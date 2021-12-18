#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os, sys
import subprocess
import threading
import shutil
import time
import re

from kaldiasr.nnet3 import KaldiNNet3OnlineModel, KaldiNNet3OnlineDecoder
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

##### PATHS #####
SEP_PATH  = '/'
ROOT_PATH = os.getcwd()
#MODELDIR  = '/home/linear/py-kaldi-asr-master/models/kaldi_01_08_2021'
#MODELDIR  = '/home/linear/py-kaldi-asr-master/models/kaldi_25_03_2021'
#MODELDIR  = '/home/linear/py-kaldi-asr-master/models/kaldi_audimus'
MODELDIR  = f'{ROOT_PATH}/models/kaldi_model_20211003'
IN_PATH   = f'{ROOT_PATH}/data/input'
PROC_PATH = f'{ROOT_PATH}/data/processing'
OUT_PATH  = f'{ROOT_PATH}/data/output'
ERR_PATH  = f'{ROOT_PATH}/data/error'
SPLIT_PATH = f'{ROOT_PATH}/data/split_info'
OUT_SPLIT_PATH = f'{ROOT_PATH}/data/split_output'

##### CONTROL #####
IN_QUEUE  = None
OBSERVER_IN_PATH = None
KALDI_MODEL = None
DECODER     = None
SUPPORTED_EXT = ['.mp3', '.mp4', '.wav']

RUN_DECODER = True
RUN_OBSERVER = True

##### SPLIT #####
SPLIT_MIDIA = True
NOISE_THREADSHOLD = -28
NOISE_INTERVAL = 0.5
SPLIT_LOG = False

##### OUTPUT FORMAT
SRT = True

def supported_ext(ext):
	try:
		return True if SUPPORTED_EXT.index(ext) >= 0 else False
	except:
		return False

def check_in_out_path():
	if not os.path.isdir(IN_PATH):
		os.mkdir(IN_PATH)

	if not os.path.isdir(PROC_PATH):
		os.mkdir(PROC_PATH)

	if not os.path.isdir(OUT_PATH):
		os.mkdir(OUT_PATH)

	if not os.path.isdir(ERR_PATH):
		os.mkdir(ERR_PATH)

	if SPLIT_MIDIA and not os.path.isdir(SPLIT_PATH):
		os.mkdir(SPLIT_PATH)

	if SPLIT_MIDIA and not os.path.isdir(OUT_SPLIT_PATH):
		os.mkdir(OUT_SPLIT_PATH)

	all_files_proc = os.listdir(PROC_PATH)
	for file in all_files_proc:
		print(">> !! Removing : " + os.path.join(PROC_PATH, file))
		os.remove(os.path.join(PROC_PATH, file))


def get_all_files(data_path):    
	list_files = os.listdir(data_path)
	all_files = list()

	for entry in list_files:
		full_path = os.path.join(data_path, entry)

		if os.path.isdir(full_path):
			all_files = all_files + get_all_files(full_path)
		else:
			name, ext = os.path.splitext(full_path)
			if supported_ext(ext):
				all_files.append(full_path)

	return all_files

def set_export_intel64_lib():
	#cmd = 'export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so'
	#os.system(cmd)
	process = subprocess.Popen([\
		'export', \
		'LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so'],\
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	stdout, stderr = process.communicate()

def load_queue():
	return get_all_files(IN_PATH)

def change_extension_to_wav(audio, output_file):	
	process = subprocess.Popen([\
		'ffmpeg',\
		'-y',\
		'-i',\
		audio,\
		'-ar', \
		'16000',\
		'-ac',\
		'1',\
		'-f',\
		'wav',\
		output_file],\
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	stdout, stderr = process.communicate()

def check_extension(file):
	name, ext = os.path.splitext(file)
	if ext != '.wav':
		wav_file = name + '.wav'
		change_extension_to_wav(file, wav_file)
		return wav_file
	else:
		return file

def move_file(file_at, path_to):
	name = file_at.split(SEP_PATH)[-1]
	file_to = os.path.join(path_to, name)
	shutil.move(file_at, file_to)
	return file_to

def decoding(wav, output_type):
	try:
		decoded = DECODER.decode_wav_file(wav)
	except:
		decoded = False
		print(str(sys.exc_info()))

	if decoded:
		s, l = DECODER.get_decoded_string()
		output = ''

		print ("*****************************************************************")
		print ("## Sucessfull decoded %s" % wav)
		print ("*****************************************************************")

		if output_type == 'txt':
			name, ext = os.path.splitext(wav)
			decoded_file = os.path.join(OUT_PATH, name.split(SEP_PATH)[-1] + '.txt')

			f = open(decoded_file, 'w')
			f.write(s)
			f.close()

			output = decoded_file
		elif output_type == 'srt':
			output = s

		return True, output
	else:
		print ("!! ERROR: decoding %s " % wav)
		return False, ''

def info_error(s, file_error):	
	print(s)

	file_n, ext = os.path.splitext(file_error)
	file_n = file_n.split(SEP_PATH)[-1]
	f = open(os.path.join(ERR_PATH, file_n + '.txt'), 'w')
	f.write(s)
	f.close()

def run_decoder():
	while RUN_DECODER:
		list_processing = os.listdir(PROC_PATH)

		if len(list_processing) == 0 and len(IN_QUEUE) > 0:
			# get the oldest file
			in_file = min(IN_QUEUE, key=os.path.getctime)

			print(f">> Processing {in_file}")

			proc_file = move_file(in_file, PROC_PATH)
			IN_QUEUE.remove(in_file)

			if SPLIT_MIDIA:
				proc_files = split_midia(proc_file)
				proc_files = sorted(proc_files, key=lambda f: f)
			else:
				proc_files = [proc_file]

			# Output format
			if SRT:
				output_name = os.path.splitext(os.path.basename(proc_file))[0]
				decode_to_srt(proc_files, output_name)
				
			#else:
			#	decode_to_txt(proc_files, '')
			decode_to_txt(proc_files, '')

		time.sleep(1)

def decode(proc_file, output_type):
	wav = check_extension(proc_file)
	output = ''

	if os.path.isfile(wav):
		fdecoded, output = decoding(wav, output_type)

		if fdecoded:
			#out_file = move_file(proc_file, OUT_PATH)

			# If there is not an error, remove output file
			os.remove(proc_file)
		else:
			out_file = move_file(proc_file, ERR_PATH)

	if os.path.isfile(wav):
		os.remove(wav)

	if os.path.isfile(proc_file):
		s = "!! Error to process " + proc_file
		print(s)
		move_file(proc_file, ERR_PATH)
		#info_error(s, proc_file)

	return output

def run_observer():
	OBSERVER_IN_PATH.start()
	print(f"Listening path {IN_PATH} ...")
	
	while RUN_OBSERVER:
		time.sleep(1)

	OBSERVER_IN_PATH.stop()
	OBSERVER_IN_PATH.join()


def on_created(event):
	historicalSize = -1
	while os.path.isfile(event.src_path) and historicalSize != os.path.getsize(event.src_path):
		historicalSize = os.path.getsize(event.src_path)
		time.sleep(1)
	
	name, ext = os.path.splitext(event.src_path)	
	if os.path.isfile(event.src_path) and supported_ext(ext):
		IN_QUEUE.append(event.src_path)

def split_midia(file):
	intervals = detect_intervals(file)
	
	if len(intervals) > 0:
		_f = file.split(SEP_PATH)
		n, e = os.path.splitext(_f[-1])
		#n = n.split('-')[-1]
		print ("\n>> Splitting in intervals")

		split_in_intervals(intervals, n, file)
		os.remove(file)

		print(">> Finishing splitting\n")

		return [os.path.join(OUT_SPLIT_PATH, f) for f in os.listdir(OUT_SPLIT_PATH)]
	else:
		return [file]


def detect_intervals(file):
	tmpfile = os.path.join(SPLIT_PATH, 'tmp.txt')
	if os.path.isfile(tmpfile):
		os.remove(tmpfile)

	command = (f'ffmpeg -i "{file}" -af silencedetect=noise={NOISE_THREADSHOLD}dB:d={NOISE_INTERVAL} -f null - 2> {tmpfile}')
	subprocess.call(command, shell=True)

	p = '(silence_end\: [0-9]+.[0-9]+)'
	intervals = []
	
	if os.path.isfile(tmpfile):
		f = open(tmpfile, 'r', encoding='utf-8')
		for l in f.readlines():
			m = re.search(p, l) 
			if m:
				grp = m.group(1)
				i = grp.split(':')[-1]
				intervals.append(i.strip())
		f.close()
		os.remove(tmpfile)

	return intervals

def split_in_intervals(intervals, name, file):
	tmp_file = os.path.join(SPLIT_PATH, '_tmp_file.txt')
	f = open(tmp_file, 'w')
	f.write(f"file '{file}'")
	f.close()

	process = subprocess.Popen(['ffmpeg',  '-i', file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	stdout, stderr = process.communicate()
	matches = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout.decode(), re.DOTALL).groupdict()
	h = float(matches['hours'])
	m = float(matches['minutes'])
	s = float(matches['seconds'])
	e = (h*60*60) + (m*60) + s

	intervals += [str(e)]

	total_dur = []

	# Name example: CF-2306_SPO02_2021-06-21_14281_22.03.00-22.07.00.mp3
	basename = "_".join(name.split('_')[:4])
	basetime_file = (name.split('_')[4]).split('-')[0]

	i = 0
	info = []

	for c, t in enumerate(intervals):
		e = t
		ini = time_new_format(i)
		dff = float(e)-float(i)
		dur = time_new_format(dff)

		if dff < 1:
			continue

		_i, _f = time_ini_final(basetime_file, ini, dff)
		
		n = f'{basename}_{_i}-{_f}_{c}.wav'

		output_audio = os.path.join(OUT_SPLIT_PATH, n)

		command = f'ffmpeg -y -f concat -safe 0 -protocol_whitelist "file" -i {tmp_file} -ss {ini} -t {dur} -ar 16000 -ac 1 -f wav {output_audio} -preset veryfast'
		subprocess.call(command, shell=True)

		if os.path.isfile(output_audio):
			db = noise_volume(output_audio, name)
			if db <= NOISE_THREADSHOLD:
				os.remove(output_audio)
				d = {'dur': dff, 'file': output_audio, 'status': 'REMOVED', 'ini': str(ini), 'end': time_new_format(e) }
				total_dur.append(d)
			else:
				d = {'dur': dff, 'file': output_audio, 'status': 'INCLUDED', 'ini': str(ini), 'end': time_new_format(e) }
				total_dur.append(d)

		i = e
	
	if SPLIT_LOG:
		s1 = sum([d['dur'] for d in total_dur if d['status'] == 'REMOVED'])
		s2 = sum([d['dur'] for d in total_dur if d['status'] == 'INCLUDED'])

		info.append(f'@@ REMOVED File: {file} - Total Dur: {s1}')
		for d in total_dur:
			if d['status'] == 'REMOVED':
				_o = d['file']
				_i = d['ini']
				_e = d['end']
				info.append(f'@@ -- File: {_o} - ini: {_i} - end: {_e}')
		  
		info.append(f'@@ INCLUDED File: {file} - Total Dur: {s2}')

def time_new_format(time):
	t = float(time)
	h = int(t/3600)

	t = t-(h*3600)
	m = int(t/60)

	s = t-(m*60)

	_h = str(h)
	_m = str(m)

	if len(_h) == 1: _h = "0" + _h
	if len(_m) == 1: _m = "0" + _m
	
	_s = ("{:.3f}").format(round(s, 3))
	if len(_s.split('.')[0]) == 1:
		_s = "0" + _s

	return ":".join([str(_h), str(_m), str(_s)])

def noise_volume(file, name):
	out = os.path.join(SPLIT_PATH, f'vol_{name}.txt')
	command = f'ffmpeg -i {file} -filter:a volumedetect -f null - 2> {out}'
	subprocess.call(command, shell=True)

	if os.path.isfile(out):
		f = open(out, 'r')
		v = f.read()
		f.close()

		os.remove(out)

		m = re.search('(mean_volume\: -[0-9]+.[0-9]+)', v)
		if m:
			v = m.group(1)
			v = v.split(':')[-1]
			return float(v.strip())

	return -1000

def time_ini_final(basetime_file, ini_split, dur_split):
	bt_f = basetime_file.split('.')
	s_ini = (ini_split.split('.')[0]).split(':')

	ini_h = (float(bt_f[0]) + float(s_ini[0]))*3600
	ini_m = (float(bt_f[1]) + float(s_ini[1]))*60
	ini_s = (float(bt_f[2]) + float(s_ini[2]))

	ini_ts = ini_h + ini_m + ini_s

	ti = time_new_format(ini_ts)
	tf = time_new_format(ini_ts + dur_split)

	ini = ".".join((ti[:8]).split(':'))
	final = ".".join((tf[:8]).split(':'))

	return ini, final

def decode_to_srt(proc_files, output_name):
	_srt = []

	for i, pf in enumerate(proc_files):
		ini, final = file_time_ini_final(pf)
		_srt.append(str(i+1))
		_srt.append(f'{ini},000 --> {final},000')
		text = decode(pf, 'srt')
		_srt.append(text + '\n')

		# ##########################
		# For each split save a txt
		# ##########################

		split_name = f"{'_'.join(output_name.split('_')[:4])}_{ini.replace(':', '.')}-{final.replace(':', '.')}.txt"
		decoded_file = os.path.join(OUT_PATH, split_name)

		print(f"TXT SPLIT FILE ------> {decoded_file}")

		f = open(decoded_file, 'w')
		f.write(text)
		f.close()

		# ##########################

	srt_file = os.path.join(OUT_PATH, output_name + '.srt')
	print(f"SRT JOIN FILE ------> {srt_file}")

	f = open(srt_file, 'w')
	f.write('\n'.join(_srt))
	f.close()
			

def file_time_ini_final(file):
	basename = os.path.basename(file)
	basename = "-".join(basename.split('-')[1:])
	info_arquivo = basename.split('_')

	id_configuracao = int(info_arquivo[0])
	no_arquivo = info_arquivo[4]
	no_path = SEP_PATH + SEP_PATH.join(info_arquivo[1:3]) + SEP_PATH + no_arquivo

	data = no_path.split('/')[2]

	time = no_arquivo.replace('.', ':')
	ini = time.split('-')[0]
	final = (time.split('-')[1])[:8]

	return ini, final

def decode_to_txt(proc_files, output_name):
	for pf in proc_files:
		decode(pf, 'txt')


if __name__== '__main__':
	thr_observer = None
	thr_decorder = None
	try:
		print("Starting program ...")

		check_in_out_path()
		IN_QUEUE = load_queue()
		print("Queue length: " + str(len(IN_QUEUE)))		

		#set_export_intel64_lib()

		print ("Loading model from %s ..." % MODELDIR)

		KALDI_MODEL = KaldiNNet3OnlineModel (MODELDIR)
		DECODER     = KaldiNNet3OnlineDecoder (KALDI_MODEL)

		# Creating Observer for input path
		#pattern = [".*(\.mp3|\.wav)$"]
		pattern = ["*"]
		ignore_patterns = None
		ignore_directories = False
		case_sensitive = True
		event_handler = PatternMatchingEventHandler(pattern, ignore_patterns, ignore_directories, case_sensitive)
		event_handler.on_created = on_created

		OBSERVER_IN_PATH = Observer()
		OBSERVER_IN_PATH.schedule(event_handler, IN_PATH, recursive=False)

		thr_observer = threading.Thread(target=run_observer, args=())
		thr_decorder = threading.Thread(target=run_decoder, args=())

		RUN_DECODER = True	
		RUN_OBSERVER = True	

		thr_observer.start()
		thr_decorder.start()
	except (KeyboardInterrupt):
		print("\nStopping program ...")
		
		RUN_DECODER = False	
		RUN_OBSERVER = False	
		