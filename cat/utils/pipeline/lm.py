"""Process of LM training
"""

import argparse

# fmt:off
import os
import sys
# after import common_utils, parent path of utils/ in in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from utils.pipeline.common_utils import *
# fmt:on


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("expdir", type=str, help="Experiment directory.")
    parser.add_argument("--start_stage", dest='stage_beg',
                        type=int, default=1, help="Start stage of processing. Default: 1")
    parser.add_argument("--stop_stage", dest='stage_end',
                        type=int, default=-1, help="Stop stage of processing. Default: last stage.")
    parser.add_argument("--ngpu", type=int, default=-1,
                        help="Number of GPUs to be used.")
    parser.add_argument("--silent", action="store_true",
                        help="Disable detailed messages output.")

    args = parser.parse_args()
    s_beg = args.stage_beg
    s_end = args.stage_end
    if s_end == -1:
        s_end = float('inf')

    assert s_end >= 1, f"Invalid stop stage: {s_end}"
    assert s_beg >= 1 and s_beg <= s_end, f"Invalid start stage: {s_beg}"

    cwd = os.getcwd()
    working_dir = args.expdir
    checkExist('d', working_dir)
    f_hyper = os.path.join(working_dir, F_HYPER_CONFIG)
    checkExist('f', f_hyper)
    hyper_cfg = readjson(f_hyper)
    if "env" in hyper_cfg:
        for k, v in hyper_cfg["env"].items():
            os.environ[k] = v

    if 'commit' not in hyper_cfg:
        log_commit(f_hyper)

    if args.ngpu > -1:
        set_visible_gpus(args.ngpu)

    initial_datainfo()

    ############ Stage 1 Tokenizer training ############
    if s_beg <= 1 and s_end >= 1:
        if not args.silent:
            print(sfmt.header("Stage 1 Tokenizer training"))
            fmt = sfmt(sfmt("Tokenizer training: ",
                            sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        else:
            fmt = ''

        hyper_cfg = readjson(f_hyper)
        if 'tokenizer' not in hyper_cfg:
            sys.stderr.write(
                sfmt.missing('tokenizer', raiseerror=False) +
                ", skip tokenizer training.\n"
            )
        else:
            train_tokenizer(f_hyper)

    ############ Stage 2  Pickle data ############
    if s_beg <= 2 and s_end >= 2:
        if not args.silent:
            print(sfmt.header("Stage 2 Pickle data"))
            fmt = sfmt(sfmt("Pickle data: ",
                            sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        else:
            fmt = ''
        from utils.data import pack_corpus as t2b

        hyper_cfg = readjson(f_hyper)
        assert 'data' in hyper_cfg, sfmt.missing('data', sfmt.udl(f_hyper))

        data_settings = hyper_cfg['data']
        cfg_packing = data_settings.get('packing-text-lm', {})

        assert 'tokenizer' in hyper_cfg, sfmt.missing(
            'tokenizer', sfmt.udl(f_hyper))
        cfg_packing['tokenizer'] = hyper_cfg['tokenizer']['file']

        pkldir = os.path.join(working_dir, 'pkl')
        os.makedirs(pkldir, exist_ok=True)
        # 'train' and 'dev' datasets would be merged into ones,
        # 'test' datasets would be processed individually in stage 4
        sys.stderr.write(sfmt.warn(
            "text is normalized with single thread. This might take a while.\n"
        ))
        for part in ['train', 'dev']:
            if part not in data_settings:
                sys.stderr.write(sfmt.missing(
                    part, (sfmt.udl(f_hyper), 'data'), False) +
                    ", skip.\n"
                )
                continue
            f_pkl = os.path.join(pkldir, part+'.pkl')
            if os.path.isfile(f_pkl):
                sys.stderr.write(sfmt.warn(
                    f"{sfmt.udl(f_pkl)} exists, skip.\n"
                ))
            else:
                part_text = list(get_corpus(f_hyper, part, merge=True))[0]
                spawn(t2b.main, parse_args_from_var(
                    t2b._parser(), cfg_packing,  [part_text, f_pkl]))
                os.remove(part_text)

    ############ Stage 3  NN training ############
    if s_beg <= 3 and s_end >= 3:
        if not args.silent:
            print(sfmt.header("Stage 3 NN training"))
            fmt = sfmt(sfmt("NN training: ",
                            sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        else:
            fmt = ''

        train_nn(working_dir, fmt)

    ############ Stage 4  Evaluate ############
    if s_beg <= 4 and s_end >= 4:
        # FIXME: runing script directly from NN training to decoding always producing SIGSEGV error
        if s_beg <= 3:
            os.system(" ".join([
                sys.executable,     # python interpreter
                sys.argv[0],        # file script
                working_dir,
                "--silent" if args.silent else "",
                "--start_stage=4",
                f"--stop_stage={args.stage_end}",
                f"--ngpu={args.ngpu}"
            ]))
            sys.exit(0)

        if not args.silent:
            print(sfmt.header("Stage 4 Evaluate"))
            fmt = sfmt(sfmt("Evaluate: ",
                            sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        else:
            fmt = ''

        hyper_cfg = readjson(f_hyper)
        assert 'inference' in hyper_cfg, sfmt.missing(
            'inference', sfmt.udl(f_hyper))

        cfg_infr = hyper_cfg['inference']
        checkdir = os.path.join(working_dir, D_CHECKPOINT)
        # do model averaging
        if 'avgmodel' in cfg_infr:
            checkpoint = model_average(
                setting=cfg_infr['avgmodel'],
                checkdir=checkdir,
                returnifexist=True
            )[0]
        else:
            checkpoint = None

        assert 'data' in hyper_cfg, sfmt.missing('data', sfmt.udl(f_hyper))
        assert 'test' in hyper_cfg['data'], sfmt.missing(
            'test', (sfmt.udl(f_hyper), 'data'))

        if 'inference' not in hyper_cfg:
            hyper_cfg['inference'] = {}

        infer_setting = hyper_cfg['inference']
        if 'avgmodel' in infer_setting:
            # do model averaging
            checkpoint = model_average(
                setting=infer_setting['avgmodel'],
                checkdir=checkdir,
                returnifexist=True
            )[0]
        else:
            checkpoint = None

        if 'infer' not in infer_setting:
            cfg_infr['infer'] = {}

        # defaultly compute ppl
        if 'bin' not in cfg_infr['infer']:
            cfg_infr['infer']['bin'] = 'cat.lm.ppl_compute'
        if 'option' not in cfg_infr['infer']:
            cfg_infr['infer']['option'] = {}

        infr_option = cfg_infr['infer']['option']
        intfname = cfg_infr['infer']['bin']

        # check config
        if 'config' not in infr_option:
            infr_option['config'] = os.path.join(working_dir, F_NN_CONFIG)
            checkExist('f', infr_option['config'])
        # check tokenizer
        if 'tokenizer' not in infr_option:
            assert hyper_cfg.get('tokenizer', {}).get('file', None) is not None, \
                (
                "\nyou should set at least one of:\n"
                f"1. set tokenizer:file ;\n"
                f"2. set inference:infer:option:tokenizer \n"
            )
            infr_option['tokenizer'] = hyper_cfg['tokenizer']['file']
            sys.stdout.write(fmt.format(sfmt.set(
                'tokenizer', infr_option['tokenizer']
            )))

        import importlib
        interface = importlib.import_module(intfname)
        # since the lm is likely to be n-gram one. checkpoint is None is ok.
        if 'resume' not in infr_option and checkpoint is not None:
            infr_option['resume'] = checkpoint
            sys.stdout.write(fmt.format(sfmt.set(
                'inference:infer:option:resume',
                checkpoint
            )))

        if intfname == 'cat.lm.ppl_compute':
            # we need to remove the uid in the transcript text
            # but for text resovled from local path, we assume it's raw text w/o uid.

            testsets = hyper_cfg['data']['test']
            if isinstance(testsets, str):
                sys.stdout.write('data: ' + testsets+'\n')
            else:
                sys.stdout.write('data: ' + '   '.join(testsets)+'\n')

            files_woid = list(get_corpus(f_hyper, 'test'))
            infr_option['evaluate'] = files_woid
            interface.main(parse_args_from_var(
                interface._parser(),
                infr_option,
                [infr_option['config']]
            ))
            for f in files_woid:
                os.remove(f)

        elif intfname == 'cat.lm.rescore':
            # if the input is a format string like rnnt-16_{}.nbest
            # we use it to format the test sets.
            assert 'nbestlist' in infr_option, sfmt.missing(
                'nbestlist', (sfmt.udl(f_hyper),
                              'inference', 'infer', 'option')
            )
            if infr_option.get('output', None) is None:
                suffix = os.path.basename(infr_option['nbestlist'])
                if suffix.endswith('.nbest'):
                    suffix = suffix[:-6]
                a = infr_option.get('alpha', 0)
                b = infr_option.get('beta', 0)
                if a != 0 or b != 0:
                    suffix = f"lm-a{a}b{b}_{suffix}"

                infr_option['output'] = os.path.join(
                    working_dir,
                    f"rescore/{suffix}"
                )
                os.makedirs(os.path.dirname(
                    infr_option['output']), exist_ok=True)
                if '{}' not in suffix:
                    sys.stdout.write(fmt.format(sfmt.set(
                        'inference:infer:option:output',
                        infr_option['output']
                    )))

            if '{}' in infr_option['nbestlist']:
                informatstr = True
                assert '{}' in infr_option['output'], \
                    f"you set 'nbestlist' as format string: {infr_option['nbestlist']}\n" \
                    f"... but the 'output' is not: {infr_option['output']}"

                if infr_option.get('save_lm_nbest', None) is not None and '{}' not in infr_option['save_lm_nbest']:
                    sys.stderr.write(
                        "Error:\n"
                        f"    you set 'nbestlist' as format string: {sfmt.udl(infr_option['nbestlist'])}\n"
                        f"    ... but the 'save_lm_nbest' is not: {sfmt.udl(infr_option['save_lm_nbest'])}\n"
                    )
                    sys.exit(1)
            else:
                informatstr = False

            if informatstr:
                testsets = hyper_cfg['data']['test']
                if isinstance(testsets, str):
                    testsets = [testsets]

                running_option = infr_option.copy()
                for _set in testsets:
                    for k in infr_option:
                        if isinstance(infr_option[k], str) and '{}' in infr_option[k]:
                            running_option[k] = infr_option[k].format(_set)
                            sys.stdout.write(fmt.format(f"{_set}: " + sfmt.set(
                                k, running_option[k]
                            )))
                    if os.path.isfile(running_option['output']):
                        sys.stderr.write(sfmt.warn(
                            f"{_set}: {sfmt.udl(running_option['output'])} exists, skip.\n"
                        ))
                        continue
                    interface.main(parse_args_from_var(
                        interface._parser(),
                        running_option,
                        [running_option['nbestlist'], running_option['output']]
                    ))
            else:
                if os.path.isfile(infr_option['output']):
                    sys.stderr.write(sfmt.warn(
                        f"{sfmt.udl(infr_option['output'])} exists, skip.\n"
                    ))
                else:
                    interface.main(parse_args_from_var(
                        interface._parser(),
                        infr_option,
                        [infr_option['nbestlist'], infr_option['output']]
                    ))
        else:
            sys.stderr.write(sfmt.warn(
                f"'{intfname}' only support handcrafted execution.\n"
            ))

        if 'er' in infer_setting:
            backup = readjson(f_hyper)
            new_cfg = backup.copy()
            new_cfg['inference'] = new_cfg['inference'].copy()
            if 'avgmodel' in new_cfg['inference']:
                del new_cfg['inference']['avgmodel']
            if 'infer' in new_cfg['inference']:
                del new_cfg['inference']['infer']

            try:
                dumpjson(new_cfg, f_hyper)
                os.system(" ".join([
                    sys.executable,     # python interpreter
                    os.path.join(
                        os.path.dirname(
                            sys.argv[0]
                        ),
                        'asr.py'
                    ),        # file script
                    working_dir,
                    "--silent",
                    "--start_stage=4",
                    "--stop_stage=4",
                    f"--ngpu={args.ngpu}"
                ]))

            except Exception as e:
                raise RuntimeError(str(e))
            finally:
                dumpjson(backup, f_hyper)
