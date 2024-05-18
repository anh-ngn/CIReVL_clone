# utils.py
import json
import os
from typing import Optional, Tuple, List, Dict, Union

import argparse
import torch
import tqdm

import data_utils
import prompts
from wrappers import MBLIPWrapper, SigLIPWrapper

if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32


@torch.no_grad()
def extract_image_features(device: torch.device, args: argparse.Namespace, dataset: torch.utils.data.Dataset, clip_model: SigLIPWrapper, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 8, preload: str = None, **kwargs) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    if preload is not None and os.path.exists(preload):
        print(f'Loading precomputed image features from {preload}!')
        extracted_data = pickle.load(open(preload, 'rb'))
        index_features, index_names = extracted_data['index_features'], extracted_data['index_names']
        index_ranks = [] if 'index_ranks' not in extracted_data else extracted_data['index_ranks']
        aux_data = {
        } if 'aux_data' not in extracted_data else extracted_data['aux_data']
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                             num_workers=num_workers, pin_memory=True, collate_fn=data_utils.collate_fn)

        index_features, index_names, index_ranks, aux_data = [], [], [], []
        if 'genecis' in args.dataset:
            aux_data = {'ref_features': [], 'instruct_features': []}

        try:
            print(
                f"Extracting image features {dataset.__class__.__name__} - {dataset.split}")
        except Exception as e:
            pass

        # Extract features
        index_rank = None
        for batch in tqdm.tqdm(loader):
            if 'genecis' in args.dataset:
                _, n_gallery, _, h, w = batch[3].size()
                images = batch[3].view(-1, 3, h, w)
                names, index_rank = batch[1], batch[4]
                ref_images = batch[0]
                instructions = batch[1]
            else:
                images = batch.get('image')
                names = batch.get('image_name')
                if images is None:
                    images = batch.get('reference_image')
                if names is None:
                    names = batch.get('reference_name')

            images = images.to(device)
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)
            if index_rank is not None:
                index_ranks.extend(index_rank)
            if len(aux_data):
                aux_data['ref_features'].append(
                    clip_model.encode_image(ref_images.to(device)).cpu())
                aux_data['instruct_features'].append(
                    clip_model.encode_text(instructions))

        index_features = torch.vstack(index_features)

        if 'genecis' in args.dataset:
            index_features = index_features.view(-1,
                                                 n_gallery, batch_features.size()[-1])
            index_ranks = torch.stack(index_ranks)
            aux_data['ref_features'] = torch.vstack(aux_data['ref_features'])
            aux_data['instruct_features'] = torch.vstack(
                aux_data['instruct_features'])

        if preload is not None:
            pickle.dump({'index_features': index_features, 'index_names': index_names,
                        'index_ranks': index_ranks, 'aux_data': aux_data}, open(preload, 'wb'))

    return index_features, index_names, index_ranks, aux_data


@torch.no_grad()
def generate_predictions(
    device: torch.device, args: argparse.Namespace, clip_model: SigLIPWrapper, blip_model: MBLIPWrapper, query_dataset: torch.utils.data.Dataset, preload_dict: Dict[str, Union[str, None]], **kwargs
) -> Tuple[torch.Tensor, List[str], list]:
    """
    Generates features predictions for the validation set of CIRCO
    """
    torch.cuda.empty_cache()
    batch_size = 32

    if preload_dict['captions'] is None or not os.path.exists(preload_dict['captions']):
        all_captions, relative_captions = [], []
        gt_img_ids, query_ids = [], []
        target_names, reference_names = [], []

        query_loader = torch.utils.data.DataLoader(
            dataset=query_dataset, batch_size=batch_size, num_workers=8,
            pin_memory=False, collate_fn=data_utils.collate_fn, shuffle=False)
        query_iterator = tqdm.tqdm(
            query_loader, position=0, desc='Generating image captions...')

        for batch in query_iterator:

            if 'genecis' in args.dataset:
                blip_image = batch[2].to(device)
                relative_captions.extend(batch[1])
            else:
                blip_image = batch['blip_ref_img'].to(device)
                reference_names.extend(batch['reference_name'])
                if 'fashioniq' not in args.dataset:
                    relative_captions.extend(batch['relative_caption'])
                else:
                    rel_caps = batch['relative_captions']
                    rel_caps = np.array(rel_caps).T.flatten().tolist()
                    relative_captions.extend([
                        f"{rel_caps[i].strip('.?, ')} and {rel_caps[i + 1].strip('.?, ')}" for i in range(0, len(rel_caps), 2)
                    ])

                if 'target_name' in batch:
                    target_names.extend(batch['target_name'])

                gt_key = 'gt_img_ids'
                if 'group_members' in batch:
                    gt_key = 'group_members'
                if gt_key in batch:
                    gt_img_ids.extend(np.array(batch[gt_key]).T.tolist())

                query_key = 'query_id'
                if 'pair_id' in batch:
                    query_key = 'pair_id'
                if query_key in batch:
                    query_ids.extend(batch[query_key])

            query_iterator.set_postfix_str(f'Shape: {blip_image.size()}')

            captions = []
            blip_prompt = eval(args.blip_prompt)
            for i in tqdm.trange(blip_image.size(0), position=1, desc='Iterating over batch', leave=False):
                img = blip_image[i].unsqueeze(0)
                inputs = blip_model.process(img, blip_prompt)
                caption = blip_model.generate(inputs)
                captions.append(blip_model.decode(caption)[0])
            all_captions.extend(captions)

        if preload_dict['captions'] is not None:
            res_dict = {
                'all_captions': all_captions,
                'gt_img_ids': gt_img_ids,
                'relative_captions': relative_captions,
                'target_names': target_names,
                'reference_names': reference_names,
                'query_ids': query_ids
            }
            pickle.dump(res_dict, open(preload_dict['captions'], 'wb'))
    else:
        print(
            f'Loading precomputed image captions from {preload_dict["captions"]}!')
        res_dict = pickle.load(open(preload_dict['captions'], 'rb'))
        all_captions, gt_img_ids, relative_captions, target_names, reference_names, query_ids = res_dict.values()

    # Modify Captions using LLM.
    if preload_dict['mods'] is None or not os.path.exists(preload_dict['mods']):
        modified_captions = []
        base_prompt = eval(args.llm_prompt)
        for i in tqdm.trange(len(all_captions), position=1, desc=f'Modifying captions with LLM...', leave=False):
            instruction = relative_captions[i]
            img_caption = all_captions[i]
            final_prompt = base_prompt + '\n' + "Image Content: " + img_caption
            final_prompt = final_prompt + '\n' + 'Instruction: ' + instruction
            resp = openai_api.openai_completion(
                final_prompt, engine=args.openai_engine, api_key=args.openai_key)
            # resp = llama_pipeline(final_prompt,temperature=0.6,top_p=0.9,max_length=800)[0]['generated_text']

            # extract edited description
            resp = resp.split('\n')
            description = ""
            aug = False
            for line in resp:
                if line.strip().startswith('Edited Description:'):
                    description = line.split(':')[1].strip()
                    if description == "":
                        modified_captions.append(relative_captions[i])
                    else:
                        modified_captions.append(description)
                    aug = True
                    break
            if not aug:
                modified_captions.append(relative_captions[i])

        if preload_dict['mods'] is not None:
            dump_dict = {'base_caption': all_captions,
                         'instruction': relative_captions, 'modified_captions': modified_captions}
            json.dump(dump_dict, open(preload_dict['mods'], 'w'), indent=6)
    else:
        print(
            f'Loading precomputed caption modifiers from {preload_dict["mods"]}!')
        modified_captions = json.load(open(preload_dict['mods'], 'r'))[
            'modified_captions']

    # Perform text-to-image retrieval based on the modified captions.
    predicted_features = text_encoding(
        device, clip_model, modified_captions, batch_size=batch_size, mode=args.retrieval)

    return {
        'predicted_features': predicted_features,
        'target_names': target_names,
        'targets': gt_img_ids,
        'reference_names': reference_names,
        'query_ids': query_ids,
        'start_captions': all_captions,
        'modified_captions': modified_captions,
        'instructions': relative_captions
    }


def text_encoding(device, clip_model, input_captions, batch_size=32, mode='default'):
    n_iter = int(np.ceil(len(input_captions)/batch_size))
    predicted_features = []

    for i in tqdm.trange(n_iter, position=0, desc='Encoding captions...'):
        captions_to_use = input_captions[i*batch_size:(i+1)*batch_size]
        tokenized_input_captions = clip_model.processor(
            text=captions_to_use, return_tensors="pt", padding="max_length")
        clip_text_features = clip_model.encode_text(tokenized_input_captions)
        predicted_features.append(clip_text_features)
    predicted_features = torch.vstack(predicted_features)

    return torch.nn.functional.normalize(predicted_features, dim=-1)
