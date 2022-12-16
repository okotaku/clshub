import logging
import re
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from time import time

import mmengine
import numpy as np
import pandas as pd
import torch
from mmengine import Config, DictAction, MMLogger
from mmengine.dataset import Compose, default_collate
from mmengine.runner import Runner
from mmengine.utils import get_installed_path
from modelindex.load_model_index import load
from rich.console import Console
from rich.table import Table

from mmcls.utils import register_all_modules

console = Console()
MMCLS_ROOT = Path(get_installed_path('mmcls'))


def parse_args():
    parser = ArgumentParser(description='Valid all models in model-index.yml')
    parser.add_argument('--dump', type=str, help='dump results to a yml file')
    parser.add_argument(
        '--checkpoint-root',
        help='Checkpoint file root path. If set, load checkpoint before test.')
    parser.add_argument('--img', default='demo/demo.JPEG', help='Image file')
    parser.add_argument('--models', nargs='+', help='models name to inference')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='The batch size during the inference.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference(config_file, checkpoint, work_dir, args, exp_name):
    cfg = Config.fromfile(config_file)
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint
    cfg.log_level = 'WARN'
    cfg.experiment_name = exp_name
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    test_dataset = cfg.test_dataloader.dataset
    if test_dataset.pipeline[0]['type'] != 'LoadImageFromFile':
        test_dataset.pipeline.insert(0, dict(type='LoadImageFromFile'))

    data = Compose(test_dataset.pipeline)({'img_path': args.img})
    data = default_collate([data] * args.batch_size)
    resolution = tuple(data['inputs'].shape[-2:])

    runner: Runner = Runner.from_cfg(cfg)
    model = runner.model

    # forward the model
    result = {'resolution': resolution}
    with torch.no_grad():
        time_record = []
        for _ in range(10):
            model.val_step(data)  # warmup before profiling
            torch.cuda.synchronize()
            start = time()
            model.val_step(data)
            torch.cuda.synchronize()
            time_record.append((time() - start) / args.batch_size * 1000)
        result['time_mean'] = float(np.mean(time_record[1:-1]))
        result['time_std'] = float(np.std(time_record[1:-1]))

    result['model'] = config_file.stem

    from fvcore.nn import FlopCountAnalysis, parameter_count
    with torch.no_grad():
        if hasattr(model, 'extract_feat'):
            model.forward = model.extract_feat
            model.to('cpu')
            inputs = (torch.randn((1, 3, *resolution)), )
            flops = FlopCountAnalysis(model, inputs).total()
            params = parameter_count(model)['']
            result['flops'] = int(flops)
            result['params'] = int(params)
        else:
            result['flops'] = ''
            result['params'] = ''

    return result


def show_summary(summary_data, args):
    table = Table(title='Validation Benchmark Regression Summary')
    table.add_column('Model', width=20)
    table.add_column('Validation')
    table.add_column('Resolution (h, w)')
    table.add_column('Inference Time (std) (ms/im)')
    table.add_column('Flops', justify='right', width=11)
    table.add_column('Params', justify='right')

    for model_name, summary in summary_data.items():
        row = [model_name]
        valid = summary['valid']
        color = 'green' if valid == 'PASS' else 'red'
        row.append(f'[{color}]{valid}[/{color}]')
        if valid == 'PASS':
            row.append(str(summary['resolution']))
            time_mean = f"{summary['time_mean']:.2f}"
            time_std = f"{summary['time_std']:.2f}"
            row.append(f'{time_mean}\t({time_std})'.expandtabs(8))
            row.append(str(summary['flops']))
            row.append(str(summary['params']))
        table.add_row(*row)

    console.print(table)


# Sample test whether the inference code is correct
def main(args):
    register_all_modules()
    model_index_file = MMCLS_ROOT / '.mim/model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    logger = MMLogger(
        'validation',
        logger_name='validation',
        log_file='benchmark_test_image.log',
        log_level=logging.INFO)

    summary_data = {}
    tmpdir = tempfile.TemporaryDirectory()
    for model_name, model_info in models.items():

        if model_info.config is None:
            continue

        if 'ImageNet' not in model_info.results[0].dataset:
            continue

        config = MMCLS_ROOT / f'.mim/{model_info.config}'
        config = Path(config)
        assert config.exists(), f'{model_name}: {config} not found.'

        logger.info(f'Processing: {model_name}')

        checkpoint = None

        try:
            # build the model from a config file and a checkpoint file
            result = inference(MMCLS_ROOT / config, checkpoint, tmpdir.name,
                               args, model_name)
            result['valid'] = 'PASS'
        except Exception:
            import traceback
            logger.error(f'"{config}" :\n{traceback.format_exc()}')
            result = {'valid': 'FAIL', 'model': config.stem}

        # add accuracy
        result['Top_1_Accuracy'] = model_info.results[0].metrics[
            'Top 1 Accuracy']
        result['Top_5_Accuracy'] = model_info.results[0].metrics[
            'Top 5 Accuracy']

        summary_data[model_name] = result

    tmpdir.cleanup()
    show_summary(summary_data, args)
    mmengine.dump(summary_data, args.dump)
    pd.io.json.json_normalize(summary_data.values()).sort_values(
        by='Top_1_Accuracy', ascending=False).to_csv(
            args.dump.replace('.yml', '.csv'), index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
