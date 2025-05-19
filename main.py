import torch
from utils import execute_function, get_args
# at first try train tabsyn
from tabsyn.main import main
from pathlib import Path
from tabsyn.vae.main import main as main_fn_vae
from tabsyn.main import main as main_fn_tabsyn
from tabsyn.sample_conditional import main as sample_conditional_fn_tabsyn
from tabsyn.sample import main as sample_fn_tabsyn
from pythelpers.logger import start_logging, stop_logging

script_dir = Path(__file__).parent
data_outdir = str(script_dir / 'output')
data_indir = str(script_dir / 'input')


def train_vae():
    args = get_args()
    args.dataname = 'features4ausw4linearsvc_train'
    args.datatestname = 'features4ausw4linearsvc_test'
    args.method = 'vae'
    args.mode = 'train'

    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    if not args.save_path:
        dataset_outpath = Path(data_outdir) / args.dataname / f'{args.method}.csv'
        args.save_path = str(dataset_outpath)

    args.ckpt_dir = f'{data_outdir}/{args.dataname}/ckpt/vae/'
    args.datapath = f'{data_indir}/{args.dataname}.csv'
    args.datatestpath = f'{data_indir}/{args.datatestname}.csv'

    main_fn_vae(args)


def train_tabsyn():
    args = get_args()
    args.dataname = 'features4ausw4linearsvc_train'
    args.datatestname = 'features4ausw4linearsvc_test'
    args.method = 'vae'
    args.mode = 'train'

    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    args.ckpt_dir = f'{data_outdir}/{args.dataname}/ckpt/'
    args.ckpt_dir_vae = f'{data_outdir}/{args.dataname}/ckpt/vae/'
    args.datapath = f'{data_indir}/{args.dataname}.csv'
    args.datatestpath = f'{data_indir}/{args.datatestname}.csv'

    main_fn_tabsyn(args)

def sample_tabsyn():
    args = get_args()
    args.dataname = 'features4ausw4linearsvc_train'
    args.datatestname = 'features4ausw4linearsvc_test'
    args.method = 'tabsyn3'
    args.mode = 'train'

    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'


    dataset_outpath = Path(data_outdir) / args.dataname / f'{args.method}2.csv'
    args.save_path = str(dataset_outpath)

    args.ckpt_dir = f'{data_outdir}/{args.dataname}/ckpt/'
    args.ckpt_path = f'{args.ckpt_dir}/model.pt'
    args.ckpt_dir_vae = f'{data_outdir}/{args.dataname}/ckpt/vae/'
    args.datapath = f'{data_indir}/{args.dataname}.csv'
    args.datatestpath = f'{data_indir}/{args.datatestname}.csv'
    args.vaedecoderpath = f'{args.ckpt_dir_vae}/decoder.pt'

    sample_fn_tabsyn(args)

def sample_tabsyn_conditional():
    title = 'Sample Conditional'
    descr = 'Sample Conditional, impact 0, 1 or 2'
    logger = start_logging(title, descr, ".")

    args = get_args()
    args.dataname = 'features4ausw4linearsvc_train'
    args.datatestname = 'features4ausw4linearsvc_test'
    args.method = 'tabsyn_conditional_impact0'

    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'


    dataset_outpath = Path(data_outdir) / args.dataname / f'{args.method}.csv'
    args.save_path = str(dataset_outpath)

    args.ckpt_dir = f'{data_outdir}/{args.dataname}/ckpt/'
    args.ckpt_path = f'{args.ckpt_dir}/model.pt'
    args.ckpt_dir_vae = f'{data_outdir}/{args.dataname}/ckpt/vae/'
    args.datapath = f'{data_indir}/{args.dataname}.csv'
    args.datatestpath = f'{data_indir}/{args.datatestname}.csv'
    args.vaedecoderpath = f'{args.ckpt_dir_vae}/decoder.pt'

    args.num_samples = 1000
    args.max_iterations = 1000
    args.conditions = [{'impact': '0'}, {'impact': '1'}, {'impact': '2'}]

    sample_conditional_fn_tabsyn(args)
    stop_logging(logger)


if __name__ == '__main__':
    # at first train the vae
    # train_vae()
    # after that train tabsyn
    # train_tabsyn()

    # after that sampling is possible
    sample_tabsyn()

    # custom conditional sampling of minority rows
    # sample_tabsyn_conditional()