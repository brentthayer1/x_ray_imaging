	����ڕ@����ڕ@!����ڕ@	���G��I@���G��I@!���G��I@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$����ڕ@ףp=
��?AF����+�@Y�G�z��@*	    &A2U
Iterator::Model::Map::Prefetchj�t���@!Hz�u�X@)j�t���@1Hz�u�X@:Preprocessing2K
Iterator::Model::Map%��C��@!�.���X@)5^�I�?1�d]:��?:Preprocessing2F
Iterator::Model���K��@!      Y@)����Mbp?1�.ğ�-B?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 51.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9���G��I@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ףp=
��?ףp=
��?!ףp=
��?      ��!       "      ��!       *      ��!       2	F����+�@F����+�@!F����+�@:      ��!       B      ��!       J	�G�z��@�G�z��@!�G�z��@R      ��!       Z	�G�z��@�G�z��@!�G�z��@JCPU_ONLYY���G��I@b 