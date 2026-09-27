"""
Training progress checker - runs periodically.
Usage: python check_progress.py
"""
import os, time, sys

LOG = os.path.join(os.path.dirname(__file__),
    'finetune_spfr_v1.log')

def check():
    if not os.path.exists(LOG):
        print('Log not found yet...')
        return False

    with open(LOG, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find last iteration line
    last_iter = ''
    last_epoch = ''
    for line in reversed(lines):
        if 'Ep' in line and 'Iter' in line and 'im/s' in line:
            last_iter = line.strip()
            break

    # Find epoch completion lines
    for line in reversed(lines):
        if '=== Epoch' in line and 'Complete' in line:
            last_epoch = line.strip()
            break

    ckpt_lines = [l.strip() for l in lines if 'Checkpoint saved' in l]
    
    print(f'[{time.strftime("%H:%M:%S")}]')
    if last_iter:
        print(f'  Last: {last_iter[-120:]}')
    if last_epoch:
        print(f'  {last_epoch}')
    if ckpt_lines:
        print(f'  Checkpoints: {len(ckpt_lines)} saved')

    # Check for errors
    for line in lines[-20:]:
        if 'Traceback' in line or 'Error' in line or 'OOM' in line:
            print(f'  ⚠️  ERROR: {line.strip()}')
            return False
    
    return True

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=int, default=900, help='Check interval (sec)')
    p.add_argument('--once', action='store_true')
    args = p.parse_args()

    if args.once:
        check()
    else:
        print(f'Monitoring every {args.interval}s. Ctrl+C to stop.')
        while True:
            ok = check()
            if not ok:
                print('Training may have stopped!')
            time.sleep(args.interval)
