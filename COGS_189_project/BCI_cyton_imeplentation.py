#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.1),
    on March 06, 2026, at 15:52
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

simonfei123. (2024). OpenVEP [Source code]. GitHub. https://github.com/simonfei123/OpenVEP/tree/main
        
Google. (2026). Gemini (3.1 Pro) [Large language model]. https://gemini.google.com
"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- OpenBCI/Cyton Imports ---
import glob
import time
import serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
from threading import Thread, Event
from queue import Queue
import pickle
import mne

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.1'
expName = 'test1'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- OpenBCI/Cyton Configuration Variables ---
cyton_in = True  # Set to False if no Cyton available
lsl_out = False
sampling_rate = 250
CYTON_BOARD_ID = 0  # 0 if no daisy, 2 if use daisy board, 6 if using daisy+wifi shield
BAUD_RATE = 115200
ANALOGUE_MODE = '/2'  # Reads from analog pins A5(D11), A6(D12) and if no wifi shield is present, then A7(D13) as well.
model_file_path = os.path.join(_thisDir, 'cache', 'FBTRCA_model.pkl')

# Data storage variables
eeg_data = np.zeros((8, 0))  # 8 EEG channels
aux_data = np.zeros((3, 0))  # 3 auxiliary channels (including photosensor)
timestamp_data = np.zeros((0))
eeg_trials = []
aux_trials = []
trial_ends = []
skip_count = 0

# --- OpenBCI Port Detection Function ---
def find_openbci_port():
    """Finds the port to which the Cyton Dongle is connected to."""
    # Find serial port names per OS
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
    else:
        raise EnvironmentError('Error finding ports on your operating system')
    
    openbci_port = ''
    for port in ports:
        try:
            s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b'v')
            line = ''
            time.sleep(2)
            if s.inWaiting():
                line = ''
                c = ''
                while '$$$' not in line:
                    c = s.read().decode('utf-8', errors='replace')
                    line += c
                if 'OpenBCI' in line:
                    openbci_port = port
            s.close()
        except (OSError, serial.SerialException):
            pass
    
    if openbci_port == '':
        raise OSError('Cannot find OpenBCI port.')
    else:
        return openbci_port

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\raide\\Downloads\\COGS_189_project - Copy\\COGS_189_project\\test1.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())

def init_cyton():
    """Initialize OpenBCI/Cyton board and start data stream"""
    global board, stop_event, queue_in, cyton_thread, model, eeg_data, aux_data, timestamp_data
    
    print("Initializing OpenBCI/Cyton board...")
    print(BoardShim.get_board_descr(CYTON_BOARD_ID))
    
    params = BrainFlowInputParams()
    if CYTON_BOARD_ID == -1:
        pass  # Synthetic board doesn't need a COM port
    elif CYTON_BOARD_ID != 6:
        params.serial_port = find_openbci_port()
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
    
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    
    # Configure board
    res_query = board.config_board('/0')
    print(res_query)
    res_query = board.config_board('//')
    print(res_query)
    res_query = board.config_board(ANALOGUE_MODE)
    print(res_query)
    
    board.start_stream(45000)
    stop_event = Event()
    
    def get_data(queue_in, lsl_out=False):
        while not stop_event.is_set():
            data_in = board.get_board_data()
            timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            # If using the Synthetic Board, mock the auxiliary data to prevent crashes
            if CYTON_BOARD_ID == -1:
                eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)][:8] # Keep 8 channels to match Cyton
                aux_in = np.zeros((3, len(timestamp_in))) # Create 3 rows of fake zeros for aux
            else:
                eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
                aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(timestamp_in) > 0:
                print('queue-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
                queue_in.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.1)
    
    queue_in = Queue()
    cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out))
    cyton_thread.daemon = True
    cyton_thread.start()
    
    # Load model if exists
    global model
    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
    else:
        model = None
        print("No model found at:", model_file_path)
    
    print("OpenBCI/Cyton initialization complete")
    return board, stop_event, queue_in, cyton_thread

def collect_cyton_data():
    """Collect data from Cyton queue and update global arrays"""
    global eeg_data, aux_data, timestamp_data, queue_in
    
    while not queue_in.empty():
        eeg_in, aux_in, timestamp_in = queue_in.get()
        print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
        eeg_data = np.concatenate((eeg_data, eeg_in), axis=1)
        aux_data = np.concatenate((aux_data, aux_in), axis=1)
        timestamp_data = np.concatenate((timestamp_data, timestamp_in), axis=0)

def save_cyton_data(thisExp):
    """Save all Cyton data to files"""
    global eeg_data, aux_data, eeg_trials, aux_trials
    
    # Create save directory based on experiment info
    participant = thisExp.extraInfo.get('participant', '000000')
    session = thisExp.extraInfo.get('session', '001')
    date_str = thisExp.extraInfo.get('date', data.getDateStr()).replace(':', '-').replace(' ', '_')
    
    save_dir = os.path.join(_thisDir, 'data', f'cyton_participant-{participant}', f'ses-{session}', date_str)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save data files
    np.save(os.path.join(save_dir, 'eeg_raw.npy'), eeg_data)
    np.save(os.path.join(save_dir, 'aux_raw.npy'), aux_data)
    np.save(os.path.join(save_dir, 'timestamp_raw.npy'), timestamp_data)
    
    # Save trial data if any
    if len(eeg_trials) > 0:
        np.save(os.path.join(save_dir, 'eeg_trials.npy'), np.array(eeg_trials, dtype=object))
    if len(aux_trials) > 0:
        np.save(os.path.join(save_dir, 'aux_trials.npy'), np.array(aux_trials, dtype=object))
    
    print(f"Cyton data saved to: {save_dir}")

def process_trial_data(i_trial, sampling_rate=250, stim_duration=30.0, baseline_duration=0.2):
    """Process EEG data for current trial using software timing instead of relying on the photosensor"""
    global eeg_data, aux_data, eeg_trials, aux_trials
    
    baseline_samples = int(baseline_duration * sampling_rate)
    trial_samples = int(stim_duration * sampling_rate)
    total_samples = baseline_samples + trial_samples
    
    # Make sure we actually have enough data collected to grab a full trial
    if eeg_data.shape[1] >= total_samples:
        # Grab exactly the last 30.2 seconds of data we just recorded
        trial_start = eeg_data.shape[1] - total_samples
        
        # Apply standard 2Hz - 40Hz bandpass filter
        filtered_eeg = mne.filter.filter_data(eeg_data, sfreq=sampling_rate, l_freq=2, h_freq=40, verbose=False)
        
        # Extract the exact window for this trial
        trial_eeg = np.copy(filtered_eeg[:, trial_start:])
        trial_aux = np.copy(aux_data[:, trial_start:])
        
        print(f'Trial {i_trial} sliced via software timing: ', trial_eeg.shape, trial_aux.shape)
        
        # Baseline correction (subtract the average of the 0.2s before the music)
        baseline_average = np.mean(trial_eeg[:, :baseline_samples], axis=1, keepdims=True)
        trial_eeg -= baseline_average
        
        eeg_trials.append(trial_eeg)
        aux_trials.append(trial_aux)
        
        return trial_eeg, trial_aux
    
    print(f"Warning: Not enough data collected to process trial {i_trial}")
    return None, None

def cleanup_cyton():
    """Stop Cyton stream and release board"""
    global stop_event, board
    if 'stop_event' in globals() and stop_event is not None:
        stop_event.set()
    if 'board' in globals() and board is not None:
        try:
            board.stop_stream()
            board.release_session()
        except:
            pass # Ignore if it was already released
        board = None # Prevent double-release crashes
    print("Cyton board released")

def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Initialize OpenBCI/Cyton if enabled
    if cyton_in:
        init_cyton()
        # Create photosensor dot for triggering
        from psychopy import visual
        photosensor_dot = visual.Rect(
            win=win, units='norm', width=0.05, height=0.05, 
            fillColor='white', lineWidth=0, pos=[0.95, -0.95]
        )
        photosensor_dot.color = [-1, -1, -1]  # Start with black
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "load_screen" ---
    loading = visual.TextBox2(
         win, text='Loading...', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='loading',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "control" ---
    cross2 = visual.ShapeStim(
        win=win, name='cross2', vertices='cross',
        size=(0.15, 0.15),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    # set audio backend
    sound.Sound.backend = 'ptb'
    white_noise = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='white_noise'
    )
    white_noise.setVolume(1.0)
    
    # --- Initialize components for Routine "trial" ---
    cross1 = visual.ShapeStim(
        win=win, name='cross1', vertices='cross',
        size=(0.15, 0.15),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    song = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='song'
    )
    song.setVolume(0.4)
    # Run 'Begin Experiment' code from code
    from pylsl import StreamInfo, StreamOutlet
    info = StreamInfo(name='PsychoPy_Markers', type='Markers', channel_count=1, nominal_srate=0, channel_format='int32', source_id='music_eeg')
    outlet = StreamOutlet(info)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "load_screen" ---
    # create an object to store info about Routine load_screen
    load_screen = data.Routine(
        name='load_screen',
        components=[loading],
    )
    load_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    loading.reset()
    # store start times for load_screen
    load_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    load_screen.tStart = globalClock.getTime(format='float')
    load_screen.status = STARTED
    thisExp.addData('load_screen.started', load_screen.tStart)
    load_screen.maxDuration = None
    # keep track of which components have finished
    load_screenComponents = load_screen.components
    for thisComponent in load_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "load_screen" ---
    thisExp.currentRoutine = load_screen
    load_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *loading* updates
        
        # if loading is starting this frame...
        if loading.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            loading.frameNStart = frameN  # exact frame index
            loading.tStart = t  # local t and not account for scr refresh
            loading.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(loading, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'loading.started')
            # update status
            loading.status = STARTED
            loading.setAutoDraw(True)
        
        # if loading is active this frame...
        if loading.status == STARTED:
            # update params
            pass
        
        # if loading is stopping this frame...
        if loading.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > loading.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                loading.tStop = t  # not accounting for scr refresh
                loading.tStopRefresh = tThisFlipGlobal  # on global time
                loading.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'loading.stopped')
                # update status
                loading.status = FINISHED
                loading.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=load_screen,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            load_screen.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if load_screen.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in load_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "load_screen" ---
    for thisComponent in load_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for load_screen
    load_screen.tStop = globalClock.getTime(format='float')
    load_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('load_screen.stopped', load_screen.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if load_screen.maxDurationReached:
        routineTimer.addTime(-load_screen.maxDuration)
    elif load_screen.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('COGS 189 song list_test.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # Track trial counter for Cyton data processing
    trial_counter = 0
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "control" ---
        # create an object to store info about Routine control
        control = data.Routine(
            name='control',
            components=[cross2, white_noise],
        )
        control.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        white_noise.setSound('audio/whitenoise.mp3', secs=15, hamming=True)
        white_noise.setVolume(1.0, log=False)
        white_noise.seek(0)
        # store start times for control
        control.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        control.tStart = globalClock.getTime(format='float')
        control.status = STARTED
        thisExp.addData('control.started', control.tStart)
        control.maxDuration = None
        # keep track of which components have finished
        controlComponents = control.components
        for thisComponent in control.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "control" ---
        thisExp.currentRoutine = control
        control.forceEnded = routineForceEnded = not continueRoutine
        
        if cyton_in:
            photosensor_dot.color = [1, 1, 1]  # Set the box to White
            photosensor_dot.setAutoDraw(True)  # Keep it on screen during this routine
            
        while continueRoutine and routineTimer.getTime() < 15.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross2* updates
            
            # if cross2 is starting this frame...
            if cross2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                cross2.frameNStart = frameN  # exact frame index
                cross2.tStart = t  # local t and not account for scr refresh
                cross2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross2.started')
                # update status
                cross2.status = STARTED
                cross2.setAutoDraw(True)
            
            # if cross2 is active this frame...
            if cross2.status == STARTED:
                # update params
                pass
            
            # if cross2 is stopping this frame...
            if cross2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross2.tStartRefresh + 15-frameTolerance:
                    # keep track of stop time/frame for later
                    cross2.tStop = t  # not accounting for scr refresh
                    cross2.tStopRefresh = tThisFlipGlobal  # on global time
                    cross2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross2.stopped')
                    # update status
                    cross2.status = FINISHED
                    cross2.setAutoDraw(False)
            
            # *white_noise* updates
            
            # if white_noise is starting this frame...
            if white_noise.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                white_noise.frameNStart = frameN  # exact frame index
                white_noise.tStart = t  # local t and not account for scr refresh
                white_noise.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('white_noise.started', tThisFlipGlobal)
                # update status
                white_noise.status = STARTED
                white_noise.play(when=win)  # sync with win flip
            
            # if white_noise is stopping this frame...
            if white_noise.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > white_noise.tStartRefresh + 15-frameTolerance or white_noise.isFinished:
                    # keep track of stop time/frame for later
                    white_noise.tStop = t  # not accounting for scr refresh
                    white_noise.tStopRefresh = tThisFlipGlobal  # on global time
                    white_noise.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'white_noise.stopped')
                    # update status
                    white_noise.status = FINISHED
                    white_noise.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=control,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                control.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if control.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in control.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "control" ---
        for thisComponent in control.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for control
        control.tStop = globalClock.getTime(format='float')
        control.tStopRefresh = tThisFlipGlobal
        thisExp.addData('control.stopped', control.tStop)
        white_noise.pause()  # ensure sound has stopped at end of Routine
        if cyton_in:
            photosensor_dot.setAutoDraw(False) 
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if control.maxDurationReached:
            routineTimer.addTime(-control.maxDuration)
        elif control.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-15.000000)
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[cross1, song],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        song.setSound(song_file, secs=30, hamming=True)
        song.setVolume(0.4, log=False)
        song.seek(0)
        # Run 'Begin Routine' code from code
        outlet.push_sample([int(trigger_value)])
        
        # Turn on photosensor dot at start of trial
        if cyton_in:
            photosensor_dot.color = [-1, -1, -1]  # Set the box to Black
            photosensor_dot.setAutoDraw(True)     # Keep it on screen during the music
        
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        thisExp.currentRoutine = trial
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross1* updates
            
            # if cross1 is starting this frame...
            if cross1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                cross1.frameNStart = frameN  # exact frame index
                cross1.tStart = t  # local t and not account for scr refresh
                cross1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross1.started')
                # update status
                cross1.status = STARTED
                cross1.setAutoDraw(True)
            
            # if cross1 is active this frame...
            if cross1.status == STARTED:
                # update params
                pass
            
            # if cross1 is stopping this frame...
            if cross1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross1.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    cross1.tStop = t  # not accounting for scr refresh
                    cross1.tStopRefresh = tThisFlipGlobal  # on global time
                    cross1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross1.stopped')
                    # update status
                    cross1.status = FINISHED
                    cross1.setAutoDraw(False)
            
            # *song* updates
            
            # if song is starting this frame...
            if song.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                song.frameNStart = frameN  # exact frame index
                song.tStart = t  # local t and not account for scr refresh
                song.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('song.started', tThisFlipGlobal)
                # update status
                song.status = STARTED
                song.play(when=win)  # sync with win flip
            
            # if song is stopping this frame...
            if song.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > song.tStartRefresh + 30-frameTolerance or song.isFinished:
                    # keep track of stop time/frame for later
                    song.tStop = t  # not accounting for scr refresh
                    song.tStopRefresh = tThisFlipGlobal  # on global time
                    song.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'song.stopped')
                    # update status
                    song.status = FINISHED
                    song.stop()
            
            # Check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                if cyton_in:
                    cleanup_cyton()
                endExperiment(thisExp, win=win)
                return
            
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        song.pause()  # ensure sound has stopped at end of Routine
        
        # Turn off photosensor dot at end of trial
        if cyton_in:
            photosensor_dot.setAutoDraw(False) 
            win.flip()
        
        # Collect Cyton data after trial
        if cyton_in:
            collect_cyton_data()
            # Process trial data
            trial_eeg, trial_aux = process_trial_data(
                i_trial=trial_counter,
                sampling_rate=sampling_rate,
                stim_duration=30.0,
                baseline_duration=0.2
            )
            trial_counter += 1
        
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial.maxDurationReached:
            routineTimer.addTime(-trial.maxDuration)
        elif trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)
    
    # Save Cyton data
    if cyton_in:
        save_cyton_data(thisExp)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # Cleanup Cyton
    if cyton_in:
        cleanup_cyton()
    
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)