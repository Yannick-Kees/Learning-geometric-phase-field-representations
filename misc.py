from packages import *

def report_progress(current, total, error):
    sys.stdout.write('\rProgress: {:.2%}, Current Error: {:}'.format(float(current)/total, error))
    if current==total:
       sys.stdout.write('\n')
    sys.stdout.flush()
    
