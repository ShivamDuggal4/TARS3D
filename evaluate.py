import os
import sys
import torch
import importlib
from misc import options
from misc.utils import log

log.process(os.getpid())
log.title("[{}] (evaluating...)".format(sys.argv[0]))

opt_cmd = options.parse_arguments(sys.argv[1:])
opt = options.set(opt_cmd=opt_cmd)

with torch.cuda.device(opt.device):

    trainer = importlib.import_module("trainer.{}".format(opt.model))
    t = trainer.Trainer(opt)
    t.load_dataset(opt, eval_split=opt.eval_split)
    t.build_networks(opt)
    t.restore_checkpoint(opt)
    t.setup_visualizer(opt)
    t.evaluate(opt)
