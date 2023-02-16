"""This was to test if we could exit with CTRL+C and proceed.

In fact it's possible to do this but it would be easier for us if
we just saved everything right up to that while loop so that doing
a CTRL+C can just exit code as usual."""
import signal
import time
import sys

EXIT = False

def signal_handler(sig, frame):
    global EXIT
    print('You pressed Ctrl+C!')
    print('Let us set the EXIT=True...')
    EXIT = True


# If we can use CTRL+C to exit but also to proceed.
while True and (not EXIT):
    signal.signal(signal.SIGINT, signal_handler)
    #signal.pause()

print('\nIf we can print this, that is GREAT')
for i in range(5):
    time.sleep(1)
    print(i)
