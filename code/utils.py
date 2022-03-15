import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from copy import deepcopy

def iou(x,y):
    insec = np.logical_and(x,y)
    uni = np.logical_or(x,y)
    return np.sum(insec)/(np.sum(uni))

def get_paths(img_dir, base_dir, model_code, epoch):

    paths = {
        'ann_file': os.path.join(base_dir, 'final_test_set_annotations.json'),
        'all_results_df_file': os.path.join(base_dir, f'single_model_results/all_results_df_{model_code}_{epoch}.pkl'),
        'segm_result_file': os.path.join(base_dir, f'{model_code}_segpc_val_set_results_{epoch}.segm.json'),
        'img_dir': img_dir,
        'gt_mask_dir': None
    }

    return paths

def get_best_candidate_index(cell_mask, other_masks, on_empty_intersection='none'):
    # Compute intersection of every mask with the cell mask
    intersections = [np.sum(np.logical_and(cell_mask, mask)) / np.sum(cell_mask) for mask in other_masks]

    # Even if there's no intersection we provide an index for a mask
    if on_empty_intersection == 'random':
        return np.argsort(intersections)[-1]
    # Alternatively
    elif on_empty_intersection == 'none':
        # If none of the masks intersect with the cell mask return None
        if np.sum(intersections) == 0:
            return None
        else:
            return np.argsort(intersections)[-1]

def merge_by_largest_overlap(cell_anns_df, nuc_anns_df, cyt_anns_df):
    used_nuc_indices = []
    used_cyt_indices = []
    masks_df = pd.DataFrame(columns=['nuc_mask', 'cyt_mask'])

    for row in cell_anns_df.itertuples():
        nuc_masks = list(nuc_anns_df['mask'])
        cyt_masks = list(cyt_anns_df['mask'])

        best_nuc_index = get_best_candidate_index(row.mask, nuc_masks)
        best_cyt_index = get_best_candidate_index(row.mask, cyt_masks)

        # Get best Nucleus mask
        if best_nuc_index is not None:
            used_nuc_indices += [best_nuc_index]
            result_nuc_mask = [nuc_anns_df.loc[best_nuc_index, 'mask']]
            result_nuc_segm = [nuc_anns_df.loc[best_nuc_index, 'segmentation']]
        else:
            result_nuc_mask = [np.zeros_like(row.mask)]
            result_nuc_segm = None

        # Get best Cytoplasm mask
        if best_cyt_index is not None:
            used_cyt_indices += [best_cyt_index]
            result_cyt_mask = [cyt_anns_df.loc[best_cyt_index, 'mask']]
            result_cyt_segm = [cyt_anns_df.loc[best_cyt_index, 'segmentation']]
        else:
            result_cyt_mask = [np.zeros_like(row.mask)]
            result_cyt_segm = None

        new_mask_df = pd.DataFrame({'nuc_mask': result_nuc_mask,
                                    'nuc_segm': result_nuc_segm,
                                    'cyt_mask': result_cyt_mask,
                                    'cyt_segm': result_cyt_segm}, index=[row.Index])

        masks_df = pd.concat([masks_df, new_mask_df], axis=0)

    return pd.concat([cell_anns_df, masks_df], axis=1)

def generate_anns_df(anns, initial_score_thold):
    anns_df = pd.DataFrame(anns)
    anns_df = anns_df[anns_df['score'] >= initial_score_thold]
    anns_df['mask'] = anns_df['segmentation'].map(maskUtils.decode)

    return anns_df.dropna().sort_values(by='score', ascending=False)


def get_category_annotations(result_anns, img_id, category_id, remove_empty_masks=True, initial_score_thold=0):
    category_anns = [ann for ann in result_anns if ann['image_id'] == img_id and ann['category_id'] == category_id]
    if len(category_anns) == 0:
        return None
    anns_df = generate_anns_df(category_anns, initial_score_thold)
    anns_df['not_empty'] = anns_df['mask'].apply(lambda x: np.any(x))

    if remove_empty_masks:
        anns_df = anns_df[anns_df['not_empty']].reset_index(drop=True)

    return anns_df


def merge_cell_dfs_by_largest_iou(ref_model_df, other_models_dict, iou_thold):
    # We will assign a mask per each one of the other models
    other_models_masks_df = pd.DataFrame(
        columns=list(other_models_dict.keys()) + ['nuc_' + str(mc) for mc in other_models_dict.keys()])
    already_used_indices_per_model_dict = {k: [] for k in other_models_dict.keys()}

    for row in ref_model_df.itertuples():
        result_cell_masks_dict = {k: None for k in other_models_dict.keys()}
        result_nuc_masks_dict = {'nuc_' + str(k): None for k in other_models_dict.keys()}
        for model in other_models_dict.keys():
            # if there are predictions
            if len(other_models_dict[model]) > 0:
                cell_masks = list(other_models_dict[model]['mask'])
                best_idx = get_largest_iou_index_from_segm(row.mask, cell_masks, iou_thold)
                if best_idx is not None:
                    result_cell_masks_dict[model] = [other_models_dict[model].loc[best_idx, 'mask']]
                    result_nuc_masks_dict['nuc_' + str(model)] = [other_models_dict[model].loc[best_idx, 'nuc_mask']]
                    already_used_indices_per_model_dict[model] += [best_idx]
                else:
                    result_cell_masks_dict[model] = None
                    result_nuc_masks_dict['nuc_' + str(model)] = None
            else:
                result_cell_masks_dict[model] = None
                result_nuc_masks_dict['nuc_' + str(model)] = None
        new_mask_df = pd.DataFrame({**result_cell_masks_dict, **result_nuc_masks_dict}, index=[row.Index])
        other_models_masks_df = pd.concat([other_models_masks_df, new_mask_df], axis=0)

    return pd.concat([ref_model_df, other_models_masks_df], axis=1), already_used_indices_per_model_dict


def merge_masks_majority_voting(row, other_models_codes, category):
    if category == 'cell':
        row_mask = row['mask']
    elif category == 'nuc':
        row_mask = row['nuc_mask']
    else:
        raise Exception

    if row_mask is None:
        return None
    else:
        for mc in other_models_codes:
            if category == 'cell':
                mc_mask = row[mc]
            elif category == 'nuc':
                mc_mask = row['nuc_' + mc]
            else:
                raise Exception
            if mc_mask is None:
                return None
            else:
                row_mask = np.add(row_mask, mc_mask)

        # Majority voting
        row_mask = row_mask > (len(other_models_codes) + 1) / 2

        return row_mask.astype('int')

def get_largest_iou_index_from_segm(ref_mask, other_masks, iou_thold):
    # Compute intersection of every mask with the cell mask
    ious = [iou(ref_mask,mask) for mask in other_masks]

    # If none of the masks intersect with the cell mask return None
    if np.sum(ious) == 0:
        return None
    else:
        largest_iou_idx = np.argsort(ious)[-1]
        if ious[largest_iou_idx] >= iou_thold:
            return largest_iou_idx
        else:
            return None


def process_img_for_ensemble(img_id, cocoGt, model_paths_dict, model_results_dict, ref_model_code, other_models_codes,
                             iou_thold, export_flag, output_dir):
    filename = cocoGt.loadImgs([img_id])[0]['file_name']
    orig_img = plt.imread(os.path.join(model_paths_dict[ref_model_code]['img_dir'], filename))

    print(f'img_id:{img_id}, filename:{filename}')

    # Load cell df for this img id
    ref_model_cell_df = deepcopy(
        model_results_dict[ref_model_code][model_results_dict[ref_model_code]['image_id'] == img_id].reset_index(
            drop=True))
    other_models_cell_df_dict = {mc: deepcopy(
        model_results_dict[mc][model_results_dict[mc]['image_id'] == img_id].reset_index(drop=True)) for mc in
        other_models_codes}

    # Create the final DF with all the masks to be exported
    all_instances_df = pd.DataFrame(columns=ref_model_cell_df.columns)
    already_used_indices_per_model_dict = {k: [] for k in other_models_codes}

    # if ref model has predictions for this img
    if len(ref_model_cell_df) > 0:
        # Compute the masks (they were removed for lighter storage)
        ref_model_cell_df['mask'] = ref_model_cell_df['segmentation'].apply(maskUtils.decode)
        ref_model_cell_df['nuc_mask'] = ref_model_cell_df['nuc_segm'].apply(
            lambda x: maskUtils.decode(x) if x is not None else None)

    for mc in other_models_cell_df_dict.keys():
        other_model_df = other_models_cell_df_dict[mc]
        if len(other_model_df) > 0:
            other_model_df['mask'] = other_model_df['segmentation'].apply(maskUtils.decode)
            other_model_df['nuc_mask'] = other_model_df['nuc_segm'].apply(
                lambda x: maskUtils.decode(x) if x is not None else None)
            other_models_cell_df_dict[mc] = other_model_df

    # Only if ref model has predictions for this img we merge
    if len(ref_model_cell_df) > 0:
        merged_cell_df, already_used_indices_per_model_dict = merge_cell_dfs_by_largest_iou(ref_model_cell_df,
                                                                                            other_models_cell_df_dict,
                                                                                            iou_thold)

        merged_cell_df['merged_cell_mask'] = merged_cell_df.apply(
            lambda x: merge_masks_majority_voting(x, other_models_codes, category='cell'), axis=1)
        merged_cell_df['merged_nuc_mask'] = merged_cell_df.apply(
            lambda x: merge_masks_majority_voting(x, other_models_codes, category='nuc'), axis=1)

        # Include those that were merged
        for row in merged_cell_df.iterrows():
            row_df = pd.DataFrame(row[1]).transpose()
            if (row[1].merged_cell_mask is not None) and (row[1].merged_nuc_mask is not None):
                row_df = row_df.drop(['mask', 'nuc_mask'], axis=1)
                row_df = row_df.rename(columns={'merged_cell_mask': 'mask', 'merged_nuc_mask': 'nuc_mask'})
                row_df['segmentation'] = row_df['mask'].apply(lambda x: x.astype(np.uint8)).apply(maskUtils.encode)
                row_df['nuc_segm'] = row_df['nuc_mask'].apply(lambda x: x.astype(np.uint8)).apply(maskUtils.encode)
                all_instances_df = pd.concat([all_instances_df, row_df], ignore_index=True)
            else:
                all_instances_df = pd.concat([all_instances_df, row_df], ignore_index=True)

        # Drop the innecessary columns
        all_instances_df = all_instances_df[ref_model_cell_df.columns]

    # Add the "unused" (i.e. the one that did not match for voting) from other models
    for mc in other_models_cell_df_dict.keys():
        other_model_df = other_models_cell_df_dict[mc]
        already_used_indices = already_used_indices_per_model_dict[mc]
        for row in other_model_df.iterrows():
            if row[0] in already_used_indices:
                continue
            else:
                row_df = pd.DataFrame(row[1]).transpose()
                all_instances_df = pd.concat([all_instances_df, row_df], ignore_index=True)

    # Generate images with all the instances for visualization
    all_nuc_masks = np.zeros(orig_img.shape[:2])
    for idx, (nuc_mask, cell_mask) in enumerate(zip(all_instances_df['nuc_mask'], all_instances_df['mask'])):
        if nuc_mask is None:
            nuc_mask = np.zeros_like(cell_mask)
        out_nuc_img = 20 * (np.logical_and(np.logical_not(nuc_mask), cell_mask)) + 40 * nuc_mask
        # Export individual instances
        if export_flag:
            out_nuc_img_path = os.path.join(output_dir, f'{filename[:-4]}_{idx + 1}.bmp')
            plt.imsave(out_nuc_img_path, out_nuc_img, cmap='gray', vmin=0, vmax=255)

    return all_instances_df.drop(['mask', 'nuc_mask'], axis=1)



