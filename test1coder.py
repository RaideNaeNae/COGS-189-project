import pylsl

print("Looking for your LSL stream on the network...")

# The modern LSL command to find a stream by its specific properties
streams = pylsl.resolve_byprop('name', 'PsychoPy_Markers')
inlet = pylsl.StreamInlet(streams[0])

print("Connected successfully! Waiting for music triggers...")

while True:
    sample, timestamp = inlet.pull_sample()
    if sample:
        print(f"BINGO! Caught trigger: {sample[0]}")