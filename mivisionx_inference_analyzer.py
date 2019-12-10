__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2019, AMD MIVisionX"
__credits__     = ["Mike Schmit; Hansel Yang; Lakshmi Kumar;"]
__license__     = "MIT"
__version__     = "1.0"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "Kiriti.NageshGowda@amd.com"
__status__      = "Shipping"
__script_name__ = "MIVisionX Inference Analyzer"

import argparse
import os
import sys
import ctypes
import cv2
import time
import numpy
import numpy as np
from numpy.ctypeslib import ndpointer
from inference_control import *

# global variables
FP16inference = False
verbosePrint = False
labelNames = None
colors =[
        (0,153,0),        # Top1
        (153,153,0),      # Top2
        (153,76,0),       # Top3
        (0,128,255),      # Top4
        (255,102,102),    # Top5
        ];

# AMD Neural Net python wrapper
class AnnAPI:
	def __init__(self,library, modeType):
		self.lib = ctypes.cdll.LoadLibrary(library)
		self.annQueryInference = self.lib.annQueryInference
		self.annQueryInference.restype = ctypes.c_char_p
		self.annQueryInference.argtypes = []
		self.annCreateInference = self.lib.annCreateInference
		self.annCreateInference.restype = ctypes.c_void_p
		self.annCreateInference.argtypes = [ctypes.c_char_p]
		self.annReleaseInference = self.lib.annReleaseInference
		self.annReleaseInference.restype = ctypes.c_int
		self.annReleaseInference.argtypes = [ctypes.c_void_p]
		self.annCopyToInferenceInput = self.lib.annCopyToInferenceInput
		self.annCopyToInferenceInput.restype = ctypes.c_int
		self.annCopyToInferenceInput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_bool]
		self.annCopyFromInferenceOutput = self.lib.annCopyFromInferenceOutput
		self.annCopyFromInferenceOutput.restype = ctypes.c_int
		self.annCopyFromInferenceOutput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
		self.annRunInference = self.lib.annRunInference
		self.annRunInference.restype = ctypes.c_int
		self.annRunInference.argtypes = [ctypes.c_void_p, ctypes.c_int]
		if modeType == 2:
			self.annCopyFromInferenceOutput_1 = self.lib.annCopyFromInferenceOutput_1
			self.annCopyFromInferenceOutput_1.restype = ctypes.c_int
			self.annCopyFromInferenceOutput_1.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
			self.annCopyFromInferenceOutput_2 = self.lib.annCopyFromInferenceOutput_2
			self.annCopyFromInferenceOutput_2.restype = ctypes.c_int
			self.annCopyFromInferenceOutput_2.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
		print('OK: AnnAPI found "' + self.annQueryInference().decode("utf-8") + '" as configuration in ' + library)

# classifier definition
class annieObjectWrapper():
	def __init__(self, annpythonlib, weightsfile, modeType):
		self.api = AnnAPI(annpythonlib, modeType)
		self.hdl = self.api.annCreateInference(weightsfile.encode('utf-8'))

		if modeType == 1:
			input_info,output_info,empty = self.api.annQueryInference().decode("utf-8").split(';')
			input,name,n_i,c_i,h_i,w_i = input_info.split(',')
			outputCount = output_info.split(",")
			stringcount = len(outputCount)
			if stringcount == 6:
				output,opName,n_o,c_o,h_o,w_o = output_info.split(',')
			else:
				output,opName,n_o,c_o= output_info.split(',')
				h_o = '1'
				w_o  = '1'
		elif modeType == 2:
			inp_out_list = self.api.annQueryInference().decode("utf-8").split(';')
			str_count = len(inp_out_list)
			self.out_list = []
			for i in range(str_count-1):
				if (inp_out_list[i].split(',')[0] == 'input'):
					input,name,n_i,c_i,h_i,w_i = inp_out_list[i].split(',')
				else:
					self.out_list.append([int(j) for j in inp_out_list[i].split(',')[2:]])

		if modeType == 1:
			self.dim = (int(w_i),int(h_i))
			self.outputDim = (int(n_o),int(c_o),int(h_o),int(w_o))
		elif modeType == 2:
			self.inp_dim = (int(h_i),int(w_i))
			self.num_outputs = len(self.out_list)
			self.outputs = []
			self.dim = (int(h_i),int(w_i))
			self.nms_threshold = 0.4
			self.conf_thres = 0.5
			self.num_classes = 80
			self.threshold = 0.18

	### Compute intersection of union score between bounding boxes
	def bbox_iou(self, bbox1, bbox2):
		#Get the coordinates of bounding boxes
		b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:,0], bbox1[:,1], bbox1[:,2], bbox1[:,3]
		b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:,0], bbox2[:,1], bbox2[:,2], bbox2[:,3]
        
		#get the corrdinates of the intersection rectangle
		inter_rect_x1 = np.maximum(b1_x1, b2_x1)
		inter_rect_y1 = np.maximum(b1_y1, b2_y1)
		inter_rect_x2 = np.minimum(b1_x2, b2_x2)
		inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        
		#Intersection area
		inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=None) \
                     * np.clip(inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=None)

		#Union Area
		b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
		b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
		iou = inter_area / (b1_area + b2_area - inter_area) 
		return iou
	

	### Transform the logspace offset to linear space coordinates
	### and rearrange the row-wise output
	def predict_transform(self, prediction, anchors):
		batch_size = prediction.shape[0]
		stride =  self.inp_dim[0] // prediction.shape[2]
		grid_size = self.inp_dim[0] // stride
		bbox_attrs = 5 + self.num_classes
		num_anchors = len(anchors)
        
		prediction = np.reshape(prediction, (batch_size, bbox_attrs*num_anchors, grid_size*grid_size))
		prediction = np.swapaxes(prediction, 1, 2)
		prediction = np.reshape(prediction, (batch_size, grid_size*grid_size*num_anchors, bbox_attrs))
		anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
		#Sigmoid the  centre_X, centre_Y. and object confidencce
		prediction[:,:,0] = 1 / (1 + np.exp(-prediction[:,:,0]))
		prediction[:,:,1] = 1 / (1 + np.exp(-prediction[:,:,1]))
		prediction[:,:,4] = 1 / (1 + np.exp(-prediction[:,:,4]))
        
		#Add the center offsets
		grid = np.arange(grid_size)
		a,b = np.meshgrid(grid, grid)

		x_offset = a.reshape(-1,1)
		y_offset = b.reshape(-1,1)


		x_y_offset = np.concatenate((x_offset, y_offset), 1)
		x_y_offset = np.tile(x_y_offset, (1, num_anchors))
		x_y_offset = np.expand_dims(x_y_offset.reshape(-1,2), axis=0)
		prediction[:,:,:2] += x_y_offset

		#log space transform height, width and box corner point x-y
		anchors = np.tile(anchors, (grid_size*grid_size, 1))
		anchors = np.expand_dims(anchors, axis=0)

		prediction[:,:,2:4] = np.exp(prediction[:,:,2:4])*anchors
		prediction[:,:,5: 5 + self.num_classes] = 1 / (1 + np.exp(-prediction[:,:, 5 : 5 + self.num_classes]))
		prediction[:,:,:4] *= stride

		box_corner = np.zeros(prediction.shape)
		box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
		box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
		box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
		box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
		prediction[:,:,:4] = box_corner[:,:,:4]

		return prediction

	def rects_prepare(self, output):
		prediction = None
		# transform prediction coordinates to correspond to pixel location
		for i in range(len(output)):
			# anchor sizes are borrowed from YOLOv3 config file
			if i == 0: 
				anchors = [(116, 90), (156, 198), (373, 326)] 
			elif i == 1:
				anchors = [(30, 61), (62, 45), (59, 119)]
			elif i == 2: 
				anchors = [(10, 13), (16, 30), (33, 23)]
			if prediction is None:
				prediction = self.predict_transform(self.outputs[i], anchors=anchors)
			else:
				prediction = np.concatenate([prediction, self.predict_transform(self.outputs[i], anchors=anchors)], axis=1)

		# confidence thresholding
		conf_mask = np.expand_dims((prediction[:,:,4] > self.conf_thres), axis=2)
		prediction = prediction * conf_mask
		prediction = prediction[np.nonzero(prediction[:, :, 4])]

		# rearrange results
		img_result = np.zeros((prediction.shape[0], 6))
		max_conf_cls = np.argmax(prediction[:, 5:5+self.num_classes], 1)
		#max_conf_score = np.amax(prediction[:, 5:5+num_classes], 1)

		img_result[:, :4] = prediction[:, :4]
		img_result[:, 4] = max_conf_cls
		img_result[:, 5] = prediction[:, 4]     
		#img_result[:, 5] = max_conf_score
        
		# non-maxima suppression
		result = []

		img_result = img_result[img_result[:, 5].argsort()[::-1]] 

		ind = 0
		while ind < img_result.shape[0]:
			bbox_cur = np.expand_dims(img_result[ind], 0)
			ious = self.bbox_iou(bbox_cur, img_result[(ind+1):])
			nms_mask = np.expand_dims(ious < self.nms_threshold, axis=2)
			img_result[(ind+1):] = img_result[(ind+1):] * nms_mask
			img_result = img_result[np.nonzero(img_result[:, 5])]
			ind += 1
        
		for ind in range(img_result.shape[0]):
			pt1 = [int(img_result[ind, 0]), int(img_result[ind, 1])]
			pt2 = [int(img_result[ind, 2]), int(img_result[ind, 3])]
			cls, prob = int(img_result[ind, 4]), img_result[ind, 5]
			result.append((pt1, pt2, cls, prob))

		return result
    
    ### get the mapping from index to classname 
	def get_classname_mapping(self, classfile):
		mapping = dict()
		with open(classfile, 'r') as fin:
			lines = fin.readlines()
			for ind, line in enumerate(lines):
				mapping[ind] = line.strip()
		return mapping

	def __del__(self):
		self.api.annReleaseInference(self.hdl)

	def runInference(self, img, out, modeType):
		# create input.f32 file
		img_r = img[:,:,0]
		img_g = img[:,:,1]
		img_b = img[:,:,2]
		img_t = np.concatenate((img_r, img_g, img_b), 0)	
		# copy input f32 to inference input
		status = self.api.annCopyToInferenceInput(self.hdl, np.ascontiguousarray(img_t, dtype=np.float32), (img.shape[0]*img.shape[1]*3*4), 0)
		if(status):
				print('ERROR: annCopyToInferenceInput Failed ')
		# run inference
		status = self.api.annRunInference(self.hdl, 1, modeType)
		if(status):
				print('ERROR: annRunInference Failed ')
		# copy output f32
		if modeType ==1:
			status = self.api.annCopyFromInferenceOutput(self.hdl, np.ascontiguousarray(out, dtype=np.float32), out.nbytes)
			if(status):
				print('ERROR: annCopyFromInferenceOutput Failed ')
		elif modeType == 2:
			status = self.api.annCopyFromInferenceOutput(self.hdl, np.ascontiguousarray(self.outputs[0], dtype=np.float32), self.outputs[0].nbytes)
			print('INFO: annCopyFromInferenceOutput status %d for output0' %(status))
			if self.num_outputs > 1:
				status = self.api.annCopyFromInferenceOutput_1(self.hdl, np.ascontiguousarray(self.outputs[1], dtype=np.float32), self.outputs[1].nbytes)
				print('INFO: annCopyFromInferenceOutput_1 status %d for output1' %(status))
			if self.num_outputs > 2:
				self.api.annCopyFromInferenceOutput_2(self.hdl, np.ascontiguousarray(self.outputs[2], dtype=np.float32), self.outputs[2].nbytes)
				print('INFO: annCopyFromInferenceOutput_2 status %d for output2' %(status))
		return out

	def classify(self, img, modeType):
		# create output.f32 buffer
		if modeType == 1:
			out_buf = bytearray(self.outputDim[0]*self.outputDim[1]*self.outputDim[2]*self.outputDim[3]*4)
			out = np.frombuffer(out_buf, dtype=numpy.float32)
		elif modeType == 2:
			self.outputs = []
			for i in range(self.num_outputs):
				out_buf_shape = self.out_list[i]
				out_buf_size = out_buf_shape[0]*out_buf_shape[1]*out_buf_shape[2]*out_buf_shape[3]*4
				out_buf = bytearray(out_buf_size)
				self.outputs.append(np.frombuffer(out_buf, dtype=np.float32))
				self.outputs[i] = np.reshape(self.outputs[i], out_buf_shape)
	        
		# run inference & receive output
		if modeType == 1:
			output = self.runInference(img, out, modeType)
		elif modeType == 2:
			output = self.runInference(img, self.outputs, modeType)
		return output

# process classification output function
def processClassificationOutput(inputImage, modelName, modelOutput):
	# post process output file
	start = time.time()
	softmaxOutput = np.float32(modelOutput)
	topIndex = []
	topLabels = []
	topProb = []
	for x in softmaxOutput.argsort()[-5:]:
		topIndex.append(x)
		topLabels.append(labelNames[x])
		topProb.append(softmaxOutput[x])
	end = time.time()
	if(verbosePrint):
		print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms'

	# display output
	start = time.time()
	# initialize the result image
	resultImage = np.zeros((250, 525, 3), dtype="uint8")
	resultImage.fill(255)
	cv2.putText(resultImage, 'MIVisionX Object Classification', (25,  25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
	topK = 1   
	for i in reversed(range(5)):
		txt =  topLabels[i].decode('utf-8')[:-1]
		conf = topProb[i]
		txt = 'Top'+str(topK)+':'+txt+' '+str(int(round((conf*100), 0)))+'%' 
		size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		t_height = size[0][1]
		textColor = (colors[topK - 1])
		cv2.putText(resultImage,txt,(45,t_height+(topK*30+40)),cv2.FONT_HERSHEY_SIMPLEX,0.5,textColor,1)
		topK = topK + 1
	end = time.time()
	if(verbosePrint):
		print '%30s' % 'Processed results image in ', str((end - start)*1000), 'ms'

	return resultImage, topIndex, topProb

# MIVisionX Classifier
if __name__ == '__main__':
    
	if len(sys.argv) == 1:
		app = QtGui.QApplication(sys.argv)
		panel = inference_control()
		app.exec_()
		modelFormat = (str)(panel.model_format)
		modelName = (str)(panel.model_name)
		modelLocation = (str)(panel.model)
		modelInputDims = (str)(panel.input_dims)
		modelOutputDims = (str)(panel.output_dims)
		label = (str)(panel.label)
		outputDir = (str)(panel.output)
		imageDir = (str)(panel.image)
		imageVal = (str)(panel.val)
		hierarchy = (str)(panel.hier)
		inputAdd = (str)(panel.add)
		inputMultiply = (str)(panel.multiply)
		fp16 = (str)(panel.fp16)
		replaceModel = (str)(panel.replace)
		verbose = (str)(panel.verbose)
		mode = str(1)
	else:
		parser = argparse.ArgumentParser()
		parser.add_argument('--model_format',		type=str, required=True,	help='pre-trained model format, options:caffe/onnx/nnef [required]')
		parser.add_argument('--model_name',			type=str, required=True,	help='model name                             [required]')
		parser.add_argument('--model',				type=str, required=True,	help='pre_trained model file/folder          [required]')
		parser.add_argument('--model_input_dims',	type=str, required=True,	help='c,h,w - channel,height,width           [required]')
		parser.add_argument('--model_output_dims',	type=str, required=True,	help='c,h,w - channel,height,width           [required]')
		parser.add_argument('--label',				type=str, required=True,	help='labels text file                       [required]')
		parser.add_argument('--output_dir',			type=str, required=True,	help='output dir to store ADAT results       [required]')
		parser.add_argument('--image_dir',			type=str, required=True,	help='image directory for analysis           [required]')
		parser.add_argument('--image_val',			type=str, default='',		help='image list with ground truth           [optional]')
		parser.add_argument('--hierarchy',			type=str, default='',		help='AMD proprietary hierarchical file      [optional]')
		parser.add_argument('--add',				type=str, default='', 		help='input preprocessing factor [optional - default:[0,0,0]]')
		parser.add_argument('--multiply',			type=str, default='',		help='input preprocessing factor [optional - default:[1,1,1]]')
		parser.add_argument('--fp16',				type=str, default='no',		help='quantize to FP16 			[optional - default:no]')
		parser.add_argument('--replace',			type=str, default='no',		help='replace/overwrite model   [optional - default:no]')
		parser.add_argument('--verbose',			type=str, default='no',		help='verbose                   [optional - default:no]')
		parser.add_argument('--mode',				type=str, default='1',		help='1:classification;2:YOLO_V3    [optional - default:1]')
		args = parser.parse_args()
		
		# get arguments
		modelFormat = args.model_format
		modelName = args.model_name
		modelLocation = args.model
		modelInputDims = args.model_input_dims
		modelOutputDims = args.model_output_dims
		label = args.label
		outputDir = args.output_dir
		imageDir = args.image_dir
		imageVal = args.image_val
		hierarchy = args.hierarchy
		inputAdd = args.add
		inputMultiply = args.multiply
		fp16 = args.fp16
		replaceModel = args.replace
		verbose = args.verbose
		mode = args.mode
	# set verbose print
	if(verbose != 'no'):
		verbosePrint = True

	# set fp16 inference turned on/off
	if(fp16 != 'no'):
		FP16inference = True

	# set paths
	modelCompilerPath = '/opt/rocm/mivisionx/model_compiler/python'
	ADATPath= '/opt/rocm/mivisionx/toolkit/analysis_and_visualization/classification'
	setupDir = '~/.mivisionx-inference-analyzer'
	analyzerDir = os.path.expanduser(setupDir)
	modelDir = analyzerDir+'/'+modelName+'_dir'
	nnirDir = modelDir+'/nnir-files'
	openvxDir = modelDir+'/openvx-files'
	modelBuildDir = modelDir+'/build'
	adatOutputDir = os.path.expanduser(outputDir)
	inputImageDir = os.path.expanduser(imageDir)
	trainedModel = os.path.expanduser(modelLocation)
	labelText = os.path.expanduser(label)
	hierarchyText = os.path.expanduser(hierarchy)
	imageValText = os.path.expanduser(imageVal)
	pythonLib = modelBuildDir+'/libannpython.so'
	weightsFile = openvxDir+'/weights.bin'
	finalImageResultsFile = modelDir+'/imageResultsFile.csv'

	#get mode of operation
	modeType = int(mode)
	# get input & output dims
	str_c_i, str_h_i, str_w_i = modelInputDims.split(',')
	c_i = int(str_c_i); h_i = int(str_h_i); w_i = int(str_w_i)
	str_c_o, str_h_o, str_w_o = modelOutputDims.split(',')
	c_o = int(str_c_o); h_o = int(str_h_o); w_o = int(str_w_o)

	# input pre-processing values
	Ax=[0,0,0]
	if(inputAdd != ''):
		Ax = [float(item) for item in inputAdd.strip("[]").split(',')]
	Mx=[1,1,1]
	if(inputMultiply != ''):
		Mx = [float(item) for item in inputMultiply.strip("[]").split(',')]

	# check pre-trained model
	if(not os.path.isfile(trainedModel) and modelFormat != 'nnef' ):
		print("\nPre-Trained Model not found, check argument --model\n")
		quit()

	# check for label file
	if (not os.path.isfile(labelText)):
		print("\nlabels.txt not found, check argument --label\n")
		quit()
	else:
		fp = open(labelText, 'r')
		labelNames = fp.readlines()
		fp.close()

	# MIVisionX setup
	if(os.path.exists(analyzerDir)):
		print("\nMIVisionX Inference Analyzer\n")
		# replace old model or throw error
		if(replaceModel == 'yes'):
			os.system('rm -rf '+modelDir)
		elif(os.path.exists(modelDir)):
			print("OK: Model exists")

	else:
		print("\nMIVisionX Inference Analyzer Created\n")
		os.system('(cd ; mkdir .mivisionx-inference-analyzer)')

	# Setup Text File for Demo
	if (not os.path.isfile(analyzerDir + "/setupFile.txt")):
		f = open(analyzerDir + "/setupFile.txt", "w")
		f.write(modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + mode)
		f.close()
	else:
		count = len(open(analyzerDir + "/setupFile.txt").readlines())
		if count < 10:
			with open(analyzerDir + "/setupFile.txt", "r") as fin:
				data = fin.read().splitlines(True)
				modelList = []
				for i in range(len(data)):
					modelList.append(data[i].split(';')[1])
				if modelName not in modelList:
					f = open(analyzerDir + "/setupFile.txt", "a")
					f.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + mode)
					f.close()
		else:
			with open(analyzerDir + "/setupFile.txt", "r") as fin:
				data = fin.read().splitlines(True)
			delModelName = data[0].split(';')[1]
			delmodelPath = analyzerDir + '/' + delModelName + '_dir'
			if(os.path.exists(delmodelPath)): 
				os.system('rm -rf ' + delmodelPath)
			with open(analyzerDir + "/setupFile.txt", "w") as fout:
			    fout.writelines(data[1:])
			with open(analyzerDir + "/setupFile.txt", "a") as fappend:
				fappend.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + mode)
				fappend.close()

	# Compile Model and generate python .so files
	if (replaceModel == 'yes' or not os.path.exists(modelDir)):
		os.system('mkdir '+modelDir)
		if(os.path.exists(modelDir)):
			# convert to NNIR
			if(modelFormat == 'caffe'):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/caffe_to_nnir.py '+trainedModel+' nnir-files --input-dims 1,'+modelInputDims+' )')
			elif(modelFormat == 'onnx'):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/onnx_to_nnir.py '+trainedModel+' nnir-files --input-dims 1,'+modelInputDims+' )')
			elif(modelFormat == 'nnef'):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnef_to_nnir.py '+trainedModel+' nnir-files )')
			else:
				print("ERROR: Neural Network Format Not supported, use caffe/onnx/nnef in arugment --model_format")
				quit()
			# convert the model to FP16
			if(FP16inference):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnir_update.py --convert-fp16 1 --fuse-ops 1 nnir-files nnir-files)')
				print("\nModel Quantized to FP16\n")
			# convert to openvx
			if(os.path.exists(nnirDir)):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnir_to_openvx.py nnir-files openvx-files)')
			else:
				print("ERROR: Converting Pre-Trained model to NNIR Failed")
				quit()
			
			# build model
			if(os.path.exists(openvxDir)):
				os.system('mkdir '+modelBuildDir)
			else:
				print("ERROR: Converting NNIR to OpenVX Failed")
				quit()
	os.system('(cd '+modelBuildDir+'; cmake ../openvx-files; make; ./anntest ../openvx-files/weights.bin )')
	print("\nSUCCESS: Converting Pre-Trained model to MIVisionX Runtime successful\n")
	
	#else:
		#print("ERROR: MIVisionX Inference Analyzer Failed")
		#quit()

	# opencv display window
	windowInput = "MIVisionX Inference Analyzer - Input Image"
	windowResult = "MIVisionX Inference Analyzer - Results"
	windowProgress = "MIVisionX Inference Analyzer - Progress"
	cv2.namedWindow(windowInput, cv2.WINDOW_GUI_EXPANDED)
	cv2.resizeWindow(windowInput, 800, 800)

	# create inference classifier
	classifier = annieObjectWrapper(pythonLib, weightsFile, modeType)

	# check for image val text
	totalImages = 0;
	if(imageVal == ''):
		print("\nFlow without Image Validation Text..Creating a file with no ground truths\n")
		imageList = os.listdir(inputImageDir)
		imageList.sort()
		imageValText = os.getcwd() + '/imageValTxt.txt'
		fp = open(imageValText , 'w')
		for imageFile in imageList:
			fp.write(imageFile + " -1" + "\n")

	if (not os.path.isfile(imageValText)):
		print("\nImage Validation Text not found, check argument --image_val\n")
		quit()
	else:
		fp = open(imageValText, 'r')
		imageValidation = fp.readlines()
		fp.close()
		totalImages = len(imageValidation)

	# original std out location 
	orig_stdout = sys.stdout
	# setup results output file
	sys.stdout = open(finalImageResultsFile,'w')
	print('Image File Name,Ground Truth Label,Output Label 1,Output Label 2,Output Label 3,\
    		Output Label 4,Output Label 5,Prob 1,Prob 2,Prob 3,Prob 4,Prob 5')
	sys.stdout = orig_stdout

	# process images
	correctTop5 = 0; correctTop1 = 0; wrong = 0; noGroundTruth = 0;
	for x in range(totalImages):
		if imageVal != '':
			imageFileName,grountTruth = imageValidation[x].decode("utf-8").split(' ')
			groundTruthIndex = int(grountTruth)
			imageFile = os.path.expanduser(inputImageDir+'/'+imageFileName)
		else:
			imageFileName = imageList[x]
			grountTruth = imageValidation[x].split(' ')[1]
			groundTruthIndex = int(grountTruth)
			imageFile = os.path.expanduser(inputImageDir+'/'+imageList[x])

		if (not os.path.isfile(imageFile)):
			print 'Image File - '+imageFile+' not found'
			quit()
		else:
			# read image
			start = time.time()
			frame = cv2.imread(imageFile)
			end = time.time()
			if(verbosePrint):
				print '%30s' % 'Read Image in ', str((end - start)*1000), 'ms'

			# resize and process frame
			start = time.time()
			resizedFrame = cv2.resize(frame, (w_i,h_i))
			RGBframe = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
			if(inputAdd != '' or inputMultiply != ''):
				pFrame = np.zeros(RGBframe.shape).astype('float32')
				for i in range(RGBframe.shape[2]):
					pFrame[:,:,i] = RGBframe.copy()[:,:,i] * Mx[i] + Ax[i]
				RGBframe = pFrame
			end = time.time()
			if(verbosePrint):
				print '%30s' % 'Input pre-processed in ', str((end - start)*1000), 'ms'

			# run inference
			start = time.time()
			output = classifier.classify(RGBframe, modeType)
			end = time.time()
			if(verbosePrint):
				print '%30s' % 'Executed Model in ', str((end - start)*1000), 'ms'

			if modeType == 1:
				# process output and display
				resultImage, topIndex, topProb = processClassificationOutput(resizedFrame, modelName, output)
				start = time.time()
				cv2.imshow(windowInput, frame)
				cv2.imshow(windowResult, resultImage)
				end = time.time()
				if(verbosePrint):
					print '%30s' % 'Processed display in ', str((end - start)*1000), 'ms\n'

			elif modeType == 2:
				rects = classifier.rects_prepare(output)
				mapping = classifier.get_classname_mapping(labelText)

				scaling_factor = min(1.0, float(h_i) / float(frame.shape[1]))
				#temporary fix
				topIndex = [-2,-2,-2,-2,-2]
				topProb = [-2,-2,-2,-2,-2]

				for pt1, pt2, cls, prob in rects:
					pt1[0] -= (h_i - scaling_factor*frame.shape[1])/2
					pt2[0] -= (h_i - scaling_factor*frame.shape[1])/2
					pt1[1] -= (h_i - scaling_factor*frame.shape[0])/2
					pt2[1] -= (h_i - scaling_factor*frame.shape[0])/2

					pt1[0] = np.clip(int(pt1[0] / scaling_factor), a_min=0, a_max=frame.shape[1])
					pt2[0] = np.clip(int(pt2[0] / scaling_factor), a_min=0, a_max=frame.shape[1])
					pt1[1] = np.clip(int(pt1[1] / scaling_factor), a_min=0, a_max=frame.shape[1])
					pt2[1] = np.clip(int(pt2[1] / scaling_factor), a_min=0, a_max=frame.shape[1])

					label = "{}:{:.2f}".format(mapping[cls], prob)
					color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

					cv2.rectangle(frame, tuple(pt1), tuple(pt2), color, 1)
					t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
					pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
					cv2.rectangle(frame, tuple(pt1), tuple(pt2), color, -1)
					cv2.putText(frame, label, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
			                    cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)
				cv2.imshow(windowInput, frame)
				
			key = cv2.waitKey(150)
			if key == 27: 
				break
			elif key == 32:
				newKey = cv2.waitKey(0)
				if newKey == 32:
					continue

			# write image results to a file
			start = time.time()
			sys.stdout = open(finalImageResultsFile,'a')
			print(imageFileName+','+str(groundTruthIndex)+','+str(topIndex[4])+
				','+str(topIndex[3])+','+str(topIndex[2])+','+str(topIndex[1])+','+str(topIndex[0])+','+str(topProb[4])+
				','+str(topProb[3])+','+str(topProb[2])+','+str(topProb[1])+','+str(topProb[0]))
			sys.stdout = orig_stdout
			end = time.time()
			if(verbosePrint):
				print '%30s' % 'Image result saved in ', str((end - start)*1000), 'ms'

			# create progress image
			start = time.time()
			progressImage = np.zeros((400, 500, 3), dtype="uint8")
			progressImage.fill(255)
			cv2.putText(progressImage, 'Inference Analyzer Progress', (25,  25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
			size = cv2.getTextSize(modelName, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
			t_width = size[0][0]
			t_height = size[0][1]
			headerX_start = int(250 -(t_width/2))
			cv2.putText(progressImage,modelName,(headerX_start,t_height+(20+40)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
			txt = 'Processed: '+str(x+1)+' of '+str(totalImages)
			size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
			cv2.putText(progressImage,txt,(50,t_height+(60+40)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
			# progress bar
			cv2.rectangle(progressImage, (50,150), (450,180), (192,192,192), -1)
			progressWidth = int(50+ ((400*(x+1))/totalImages))
			cv2.rectangle(progressImage, (50,150), (progressWidth,180), (255,204,153), -1)
			percentage = int(((x+1)/float(totalImages))*100)
			pTxt = 'progress: '+str(percentage)+'%'
			cv2.putText(progressImage,pTxt,(175,170),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

			if(groundTruthIndex == topIndex[4]):
				correctTop1 = correctTop1 + 1
				correctTop5 = correctTop5 + 1
			elif(groundTruthIndex == topIndex[3] or groundTruthIndex == topIndex[2] or groundTruthIndex == topIndex[1] or groundTruthIndex == topIndex[0]):
				correctTop5 = correctTop5 + 1
			elif(groundTruthIndex == -1):
				noGroundTruth = noGroundTruth + 1
			else:
				wrong = wrong + 1

			# top 1 progress
			cv2.rectangle(progressImage, (50,200), (450,230), (192,192,192), -1)
			progressWidth = int(50 + ((400*correctTop1)/totalImages))
			cv2.rectangle(progressImage, (50,200), (progressWidth,230), (0,153,0), -1)
			percentage = int((correctTop1/float(totalImages))*100)
			pTxt = 'Top1: '+str(percentage)+'%'
			cv2.putText(progressImage,pTxt,(195,220),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
			# top 5 progress
			cv2.rectangle(progressImage, (50,250), (450,280), (192,192,192), -1)
			progressWidth = int(50+ ((400*correctTop5)/totalImages))
			cv2.rectangle(progressImage, (50,250), (progressWidth,280), (0,255,0), -1)
			percentage = int((correctTop5/float(totalImages))*100)
			pTxt = 'Top5: '+str(percentage)+'%'
			cv2.putText(progressImage,pTxt,(195,270),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
			# wrong progress
			cv2.rectangle(progressImage, (50,300), (450,330), (192,192,192), -1)
			progressWidth = int(50+ ((400*wrong)/totalImages))
			cv2.rectangle(progressImage, (50,300), (progressWidth,330), (0,0,255), -1)
			percentage = int((wrong/float(totalImages))*100)
			pTxt = 'Mismatch: '+str(percentage)+'%'
			cv2.putText(progressImage,pTxt,(175,320),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
			# no ground truth progress
			cv2.rectangle(progressImage, (50,350), (450,380), (192,192,192), -1)
			progressWidth = int(50+ ((400*noGroundTruth)/totalImages))
			cv2.rectangle(progressImage, (50,350), (progressWidth,380), (0,255,255), -1)
			percentage = int((noGroundTruth/float(totalImages))*100)
			pTxt = 'Ground Truth unavailable: '+str(percentage)+'%'
			cv2.putText(progressImage,pTxt,(125,370),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
			
			cv2.imshow(windowProgress, progressImage)
			end = time.time()
			if(verbosePrint):
				print '%30s' % 'Progress image created in ', str((end - start)*1000), 'ms'
	# Inference Analyzer Successful
	print("\nSUCCESS: Images Inferenced with the Model\n")
	cv2.destroyWindow(windowInput)
	if modeType == 1:
		cv2.destroyWindow(windowResult)

	# Create ADAT folder and file
	print("\nADAT tool called to create the analysis toolkit\n")
	if(not os.path.exists(adatOutputDir)):
		os.system('mkdir ' + adatOutputDir)
	
	if(hierarchy == ''):
		os.system('python '+ADATPath+'/generate-visualization.py --inference_results '+finalImageResultsFile+
		' --image_dir '+inputImageDir+' --label '+labelText+' --model_name '+modelName+' --output_dir '+adatOutputDir+' --output_name '+modelName+'-ADAT')
	else:
		os.system('python '+ADATPath+'/generate-visualization.py --inference_results '+finalImageResultsFile+
		' --image_dir '+inputImageDir+' --label '+labelText+' --hierarchy '+hierarchyText+' --model_name '+modelName+' --output_dir '+adatOutputDir+' --output_name '+modelName+'-ADAT')
	print("\nSUCCESS: Image Analysis Toolkit Created\n")
	print("Press ESC to exit or close progess window\n")

	# Wait to quit
	while True:
		key = cv2.waitKey(2)
		if key == 27:
			cv2.destroyAllWindows()
			break        
		if cv2.getWindowProperty(windowProgress,cv2.WND_PROP_VISIBLE) < 1:        
			break

	outputHTMLFile = os.path.expanduser(adatOutputDir+'/'+modelName+'-ADAT-toolKit/index.html')
	os.system('firefox '+outputHTMLFile)
