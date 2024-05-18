# main.py
import os
import argparse
import torch
import termcolor

import utils
import prompts
import datasets
import data_utils
import compute_results
from wrappers import MBLIPWrapper, SigLIPWrapper

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    # Load Input Arguments.
    parser = argparse.ArgumentParser()
    # Base Arguments
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU ID to use.")
    parser.add_argument("--preload", nargs='+', type=str, default=['img_features', 'captions', 'mods'],
                        help='List of properties to preload is computed once before.')
    # Base Model Choices
    parser.add_argument("--clip", type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50x4', 'ViT-bigG-14',
                                 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14'],
                        help="Which CLIP text-to-image retrieval model to use"),
    parser.add_argument("--blip", type=str, default='blip2_t5', choices=['blip2_t5'],
                        help="BLIP Image Caption Model to use.")
    # Dataset Arguments ['dress', 'toptee', 'shirt']
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['cirr', 'circo',
                                 'fashioniq_dress', 'fashioniq_toptee', 'fashioniq_shirt',
                                 'genecis_change_attribute', 'genecis_change_object', 'genecis_focus_attribute', 'genecis_focus_object'],
                        help="Dataset to use")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
                        help='Dataset split to evaluate on. Some datasets require special testing protocols s.a. cirr/circo.')
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("--weight-path", type=str, default='',
                        help='Where to store OpenCLIP weights.')
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    # LLM & BLIP Prompt Arguments.
    available_prompts = [
        f'prompts.{x}' for x in prompts.__dict__.keys() if '__' not in x]
    parser.add_argument("--llm_prompt", default='prompts.simple_modifier_prompt', type=str, choices=available_prompts,
                        help='Denotes the base prompt to use to probe the LLM. Has to be available in prompts.py')
    parser.add_argument("--blip_prompt", default='prompts.blip_prompt', type=str, choices=available_prompts,
                        help='Denotes the base prompt to use alongside BLIP. Has to be available in prompts.py')
    parser.add_argument("--openai_engine", default='gpt-3.5-turbo', type=str, choices=['gpt-3.5-turbo', 'gpt-4'],
                        help='Openai LLM Engine to use.')
    parser.add_argument("--openai_key", default="<your_openai_key_here>", type=str,
                        help='Account key for openai LLM usage.')
    # Text-to-Image Retrieval Arguments.
    parser.add_argument("--retrieval", type=str, default='default', choices=['default'],
                        help='Type of T2I Retrieval method.')
    args = parser.parse_args()

    # Set Device.
    termcolor.cprint(
        f'Starting evaluation on {args.dataset.upper()} (split: {args.split})\n', color='green', attrs=['bold'])
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Argument Checks.
    preload_dict = {key: None for key in ['img_features', 'captions', 'mods']}
    preload_str = f'{args.dataset}_{args.blip}_{args.clip}_{args.split}'.replace(
        '/', '-')

    if len(args.preload):
        os.makedirs('precomputed', exist_ok=True)
    if 'img_features' in args.preload:
        preload_dict['img_features'] = os.path.join(
            'precomputed', preload_str + '_img_features.pkl')

    if 'captions' in args.preload:
        caption_load_str = f'{args.dataset}_{args.blip}_{args.split}'.replace(
            '/', '-')
        if args.blip_prompt != 'prompts.blip_prompt':
            preload_dict['captions'] = os.path.join(
                'precomputed', caption_load_str + f'_captions_{args.blip_prompt.split(".")[-1]}.pkl')
        else:
            preload_dict['captions'] = os.path.join(
                'precomputed', caption_load_str + '_captions.pkl')

    if 'mods' in args.preload:
        mod_load_str = f'{args.dataset}_{args.blip}_{args.split}'.replace(
            '/', '-')
        preload_dict['mods'] = os.path.join(
            'precomputed', mod_load_str + f'_mods_{args.llm_prompt.split(".")[-1]}.json')
        if args.openai_engine != 'gpt-3.5-turbo':
            preload_dict['mods'] = preload_dict['mods'].replace(
                '.json', f'_{args.openai_engine}.json')

    if args.split == 'test':
        preload_dict['test'] = preload_str + \
            f'{args.blip_prompt.split(".")[-1]}_{args.llm_prompt.split(".")[-1]}_test_submission.json'

    # Load CLIP model, BLIP model & Preprocessing.
    print(f'Loading CLIP {args.clip}... ', end='')

    if args.clip in ['ViT-bigG-14', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']:
        clip_model = SigLIPWrapper()
    else:
        clip_model = SigLIPWrapper()

    print('Done.')

    if args.preprocess_type == 'targetpad':
        print('Target pad preprocess pipeline is used.')
        preprocess = data_utils.targetpad_transform(
            1.25, 224)  # Update the size to match SigLIP
    elif args.preprocess_type == 'clip':
        print('CLIP preprocess pipeline is used.')
        preprocess = data_utils.targetpad_transform(
            1.25, 224)  # Update the size to match SigLIP

    blip_model = None
    if preload_dict['captions'] is None or not os.path.exists(preload_dict['captions']):
        blip_model = MBLIPWrapper()
    else:
        blip_model = MBLIPWrapper()
        print(f'Skipped loading of BLIP ({args.blip}).')

    # Load Evaluation Datasets.
    target_datasets, query_datasets, pairings = [], [], []

    if 'fashioniq' in args.dataset.lower():
        dress_type = args.dataset.split('_')[-1]
        target_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [
                               dress_type], 'classic', preprocess, blip_transform=preprocess))
        query_datasets.append(datasets.FashionIQDataset(args.dataset_path, args.split, [
                              dress_type], 'relative', preprocess, blip_transform=preprocess))
        pairings.append(dress_type)
        compute_results_function = compute_results.fiq

    elif args.dataset.lower() == 'cirr':
        split = 'test1' if args.split == 'test' else args.split
        target_datasets.append(datasets.CIRRDataset(
            args.dataset_path, split, 'classic', preprocess, blip_transform=preprocess))
        query_datasets.append(datasets.CIRRDataset(
            args.dataset_path, split, 'relative', preprocess, blip_transform=preprocess))
        compute_results_function = compute_results.cirr
        pairings.append('default')

    elif args.dataset.lower() == 'circo':
        target_datasets.append(datasets.CIRCODataset(
            args.dataset_path, args.split, 'classic', preprocess, blip_transform=preprocess))
        query_datasets.append(datasets.CIRCODataset(
            args.dataset_path, args.split, 'relative', preprocess, blip_transform=preprocess))
        compute_results_function = compute_results.circo
        pairings.append('default')

    elif 'genecis' in args.dataset.lower():
        prop_file = '_'.join(args.dataset.lower().split('_')[1:])
        prop_file = os.path.join(
            args.dataset_path, 'genecis', prop_file + '.json')

        if 'object' in args.dataset.lower():
            datapath = os.path.join(args.dataset_path, 'coco2017', 'val2017')
            genecis_dataset = datasets.COCOValSubset(
                root_dir=datapath, val_split_path=prop_file, transform=preprocess, blip_transform=preprocess)
        elif 'attribute' in args.dataset.lower():
            datapath = os.path.join(
                args.dataset_path, 'Visual_Genome', 'VG_All')
            genecis_dataset = datasets.VAWValSubset(
                image_dir=datapath, val_split_path=prop_file, transform=preprocess, blip_transform=preprocess)

        target_datasets.append(genecis_dataset)
        query_datasets.append(genecis_dataset)
        compute_results_function = compute_results.genecis
        pairings.append('default')

    # Evaluate performances.
    for query_dataset, target_dataset, pairing in zip(query_datasets, target_datasets, pairings):
        termcolor.cprint(
            f'\n------ Evaluating Retrieval Setup: {pairing}', color='yellow', attrs=['bold'])

        # General Input Arguments.
        input_kwargs = {
            'args': args, 'query_dataset': query_dataset, 'target_dataset': target_dataset, 'clip_model': clip_model,
            'blip_model': blip_model, 'preprocess': preprocess, 'device': device, 'split': args.split,
            'blip_transform': preprocess, 'preload_dict': preload_dict,
        }

        # Compute Target Image Features
        print(f'Extracting target image features using CLIP: {args.clip}.')
        index_features, index_names, index_ranks, aux_data = utils.extract_image_features(
            device, args, target_dataset, clip_model, preload=preload_dict['img_features'])
        index_features = torch.nn.functional.normalize(
            index_features.float(), dim=-1)
        input_kwargs.update({'index_features': index_features,
                            'index_names': index_names, 'index_ranks': index_ranks})

        # Compute Method-specific Query Features.
        print(
            f'Generating conditional query predictions (CLIP: {args.clip}, BLIP: {args.blip}).')
        out_dict = utils.generate_predictions(**input_kwargs)
        input_kwargs.update(out_dict)

        # Compute Dataset-specific Retrieval Scores.
        print('Computing final retrieval metrics.')
        if args.dataset == 'genecis_focus_attribute':
            aux_data['ref_features'] = torch.nn.functional.normalize(
                aux_data['ref_features'].float().to(device))
            out_dict['predicted_features'] = torch.nn.functional.normalize(
                (out_dict['predicted_features'].float() + aux_data['ref_features'])/2, dim=-1)

        input_kwargs.update(out_dict)
        result_metrics = compute_results_function(**input_kwargs)

        # Print metrics.
        print('\n')
        if result_metrics is not None:
            termcolor.cprint(
                f'Metrics for {args.dataset.upper()} ({args.split})- {pairing}', attrs=['bold'])
            for k, v in result_metrics.items():
                print(f"{pairing}_{k} = {v:.2f}")
        else:
            termcolor.cprint(
                f'No explicit metrics available for {args.dataset.upper()} ({args.split}) - {pairing}.', attrs=['bold'])


if __name__ == '__main__':
    main()
