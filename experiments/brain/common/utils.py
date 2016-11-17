import tensorflow as tf
import math
import csv
import os

TRACK_VARS=False
episode_stats = {'names': [], 'variables' : []}
output_dir = ''
cur_episode = -1
csv_writer = None
def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.scalar_summary('stddev/' + name, stddev)
		stat_max = tf.reduce_max(var)
		stat_min = tf.reduce_min(var)
		episode_stats['names'].extend([name+'_mean', name+'_stddev', name+'_max', name+'_min'])
		episode_stats['variables'].extend([mean, stddev, stat_max, stat_min])
		tf.scalar_summary('max/' + name, stat_max)
		tf.scalar_summary('min/' + name, stat_min)
		tf.histogram_summary(name, var)
def set_output_dir(directory):
	global output_dir
	# make dir if it doesn't exist
	if not os.path.exists(directory):
		os.makedirs(directory)
	output_dir = directory

def write_row(episode, timestep, row):
	"""
	Writes the current values for episode stats
	as a new row for the data
	"""
	# check what the current file buffer is open
	global cur_episode, output_dir
	if cur_episode != episode:
		cur_episode = episode
		# TODO test to make sure episode{} doesn't exist

		filename = '{}episode{}.csv'.format(output_dir,episode)
		if os.path.exists(filename):
			raise ValueError('{} already exists'.format(filename))
		with open(filename, 'a+') as f:
			csv_writer = csv.writer(f)
			titles = ['timestep'] + episode_stats['names']
			newrow = [timestep] + row
			csv_writer.writerow(titles)
			csv_writer.writerow(newrow)

	else:
		with open('{}episode{}.csv'.format(output_dir,episode), 'a') as f:
			csv_writer = csv.writer(f)
			csv_writer.writerow([timestep] + row)





	# open the file buffer
	# write the line
	# close the file buffer
def variable(shape,f, name="Variable"):
	"""
	Creates a tensor of SHAPE drawn from
	a random uniform distribution in 0 +/- 1/sqrt(f)
	"""
	#TODO: fix this. currently shape is a [Dimension, int] object
	v =  tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)), name=name)
	if TRACK_VARS: variable_summaries(var, name)
	return v
