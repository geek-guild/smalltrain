import os
import re
import argparse

def get_proc_id_with_pattern(pattern='.*python.*\"train_id\": \"{train_id}\"'):
    p = re.compile(pattern)

    for dirname in os.listdir('/proc'):
        if dirname == 'curproc':
            continue

        try:
            with open('/proc/{}/cmdline'.format(dirname), mode='rb') as fd:
                cmdline = fd.read().decode()
                # print('cmdline:{}'.format(cmdline))

                if p.match(cmdline) is None: continue
                proc_id = int(dirname)
                return proc_id
        except Exception:
            continue
    # proc not found
    return None

def get_proc_id_with_train_id(train_id):
    pattern = '.*python.*\"train_id\": \"{}\"'.format(train_id)
    return get_proc_id_with_pattern(pattern)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='proc')
    parser.add_argument('--train_id', '-tid', type=str, default=None, required=True,
                        help='train_id')
    args = parser.parse_args()
    print('args:{}'.format(args))
    proc_id = get_proc_id_with_train_id(train_id=args.train_id)
    print('proc_id:{}'.format(proc_id))