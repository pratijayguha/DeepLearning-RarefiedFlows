χ
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-rc2-17-ge5bf8de8��
v
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*
shared_namelayer1/kernel
o
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes

:	*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:	*
dtype0
v
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*
shared_namelayer2/kernel
o
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes

:		*
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:	*
dtype0
�
Final_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*#
shared_nameFinal_layer/kernel
y
&Final_layer/kernel/Read/ReadVariableOpReadVariableOpFinal_layer/kernel*
_output_shapes

:	*
dtype0
x
Final_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameFinal_layer/bias
q
$Final_layer/bias/Read/ReadVariableOpReadVariableOpFinal_layer/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
z
layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	* 
shared_namelayer1/kernel/m
s
#layer1/kernel/m/Read/ReadVariableOpReadVariableOplayer1/kernel/m*
_output_shapes

:	*
dtype0
r
layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namelayer1/bias/m
k
!layer1/bias/m/Read/ReadVariableOpReadVariableOplayer1/bias/m*
_output_shapes
:	*
dtype0
z
layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		* 
shared_namelayer2/kernel/m
s
#layer2/kernel/m/Read/ReadVariableOpReadVariableOplayer2/kernel/m*
_output_shapes

:		*
dtype0
r
layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namelayer2/bias/m
k
!layer2/bias/m/Read/ReadVariableOpReadVariableOplayer2/bias/m*
_output_shapes
:	*
dtype0
�
Final_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*%
shared_nameFinal_layer/kernel/m
}
(Final_layer/kernel/m/Read/ReadVariableOpReadVariableOpFinal_layer/kernel/m*
_output_shapes

:	*
dtype0
|
Final_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameFinal_layer/bias/m
u
&Final_layer/bias/m/Read/ReadVariableOpReadVariableOpFinal_layer/bias/m*
_output_shapes
:*
dtype0
z
layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	* 
shared_namelayer1/kernel/v
s
#layer1/kernel/v/Read/ReadVariableOpReadVariableOplayer1/kernel/v*
_output_shapes

:	*
dtype0
r
layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namelayer1/bias/v
k
!layer1/bias/v/Read/ReadVariableOpReadVariableOplayer1/bias/v*
_output_shapes
:	*
dtype0
z
layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		* 
shared_namelayer2/kernel/v
s
#layer2/kernel/v/Read/ReadVariableOpReadVariableOplayer2/kernel/v*
_output_shapes

:		*
dtype0
r
layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namelayer2/bias/v
k
!layer2/bias/v/Read/ReadVariableOpReadVariableOplayer2/bias/v*
_output_shapes
:	*
dtype0
�
Final_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*%
shared_nameFinal_layer/kernel/v
}
(Final_layer/kernel/v/Read/ReadVariableOpReadVariableOpFinal_layer/kernel/v*
_output_shapes

:	*
dtype0
|
Final_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameFinal_layer/bias/v
u
&Final_layer/bias/v/Read/ReadVariableOpReadVariableOpFinal_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
� 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
lm9m:m;m<m=m>v?v@vAvBvCvD
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
�
	variables
layer_regularization_losses

layers
non_trainable_variables
trainable_variables
regularization_losses
 metrics
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
!layer_regularization_losses

"layers
#non_trainable_variables
trainable_variables
regularization_losses
$metrics
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
%layer_regularization_losses

&layers
'non_trainable_variables
trainable_variables
regularization_losses
(metrics
^\
VARIABLE_VALUEFinal_layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEFinal_layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
)layer_regularization_losses

*layers
+non_trainable_variables
trainable_variables
regularization_losses
,metrics
 

0
1
2
3
 

-0
 
 
 
 
 
 
 
 
 
 
 
 
x
	.total
	/count
0
_fn_kwargs
1	variables
2trainable_variables
3regularization_losses
4	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1
 
 
�
1	variables
5layer_regularization_losses

6layers
7non_trainable_variables
2trainable_variables
3regularization_losses
8metrics
 
 

.0
/1
 
wu
VARIABLE_VALUElayer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUElayer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUElayer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUElayer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEFinal_layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEFinal_layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUElayer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUElayer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUElayer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUElayer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEFinal_layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEFinal_layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_InputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputlayer1/kernellayer1/biaslayer2/kernellayer2/biasFinal_layer/kernelFinal_layer/bias*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*.
f)R'
%__inference_signature_wrapper_1147850
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp&Final_layer/kernel/Read/ReadVariableOp$Final_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp#layer1/kernel/m/Read/ReadVariableOp!layer1/bias/m/Read/ReadVariableOp#layer2/kernel/m/Read/ReadVariableOp!layer2/bias/m/Read/ReadVariableOp(Final_layer/kernel/m/Read/ReadVariableOp&Final_layer/bias/m/Read/ReadVariableOp#layer1/kernel/v/Read/ReadVariableOp!layer1/bias/v/Read/ReadVariableOp#layer2/kernel/v/Read/ReadVariableOp!layer2/bias/v/Read/ReadVariableOp(Final_layer/kernel/v/Read/ReadVariableOp&Final_layer/bias/v/Read/ReadVariableOpConst*!
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__traced_save_1148057
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biasFinal_layer/kernelFinal_layer/biastotalcountlayer1/kernel/mlayer1/bias/mlayer2/kernel/mlayer2/bias/mFinal_layer/kernel/mFinal_layer/bias/mlayer1/kernel/vlayer1/bias/vlayer2/kernel/vlayer2/bias/vFinal_layer/kernel/vFinal_layer/bias/v* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__traced_restore_1148129��
�	
�
C__inference_layer2_layer_call_and_return_conditional_losses_1147741

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������	2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
'__inference_model_layer_call_fn_1147838	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_11478292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameInput
�
�
H__inference_Final_layer_layer_call_and_return_conditional_losses_1147966

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
C__inference_layer2_layer_call_and_return_conditional_losses_1147949

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������	2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
C__inference_layer1_layer_call_and_return_conditional_losses_1147931

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������	2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
"__inference__wrapped_model_1147703	
input/
+model_layer1_matmul_readvariableop_resource0
,model_layer1_biasadd_readvariableop_resource/
+model_layer2_matmul_readvariableop_resource0
,model_layer2_biasadd_readvariableop_resource4
0model_final_layer_matmul_readvariableop_resource5
1model_final_layer_biasadd_readvariableop_resource
identity��(model/Final_layer/BiasAdd/ReadVariableOp�'model/Final_layer/MatMul/ReadVariableOp�#model/layer1/BiasAdd/ReadVariableOp�"model/layer1/MatMul/ReadVariableOp�#model/layer2/BiasAdd/ReadVariableOp�"model/layer2/MatMul/ReadVariableOp�
"model/layer1/MatMul/ReadVariableOpReadVariableOp+model_layer1_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02$
"model/layer1/MatMul/ReadVariableOp�
model/layer1/MatMulMatMulinput*model/layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
model/layer1/MatMul�
#model/layer1/BiasAdd/ReadVariableOpReadVariableOp,model_layer1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02%
#model/layer1/BiasAdd/ReadVariableOp�
model/layer1/BiasAddBiasAddmodel/layer1/MatMul:product:0+model/layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
model/layer1/BiasAdd
model/layer1/ReluRelumodel/layer1/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
model/layer1/Relu�
"model/layer2/MatMul/ReadVariableOpReadVariableOp+model_layer2_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02$
"model/layer2/MatMul/ReadVariableOp�
model/layer2/MatMulMatMulmodel/layer1/Relu:activations:0*model/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
model/layer2/MatMul�
#model/layer2/BiasAdd/ReadVariableOpReadVariableOp,model_layer2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02%
#model/layer2/BiasAdd/ReadVariableOp�
model/layer2/BiasAddBiasAddmodel/layer2/MatMul:product:0+model/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
model/layer2/BiasAdd
model/layer2/ReluRelumodel/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
model/layer2/Relu�
'model/Final_layer/MatMul/ReadVariableOpReadVariableOp0model_final_layer_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02)
'model/Final_layer/MatMul/ReadVariableOp�
model/Final_layer/MatMulMatMulmodel/layer2/Relu:activations:0/model/Final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/Final_layer/MatMul�
(model/Final_layer/BiasAdd/ReadVariableOpReadVariableOp1model_final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model/Final_layer/BiasAdd/ReadVariableOp�
model/Final_layer/BiasAddBiasAdd"model/Final_layer/MatMul:product:00model/Final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/Final_layer/BiasAdd�
IdentityIdentity"model/Final_layer/BiasAdd:output:0)^model/Final_layer/BiasAdd/ReadVariableOp(^model/Final_layer/MatMul/ReadVariableOp$^model/layer1/BiasAdd/ReadVariableOp#^model/layer1/MatMul/ReadVariableOp$^model/layer2/BiasAdd/ReadVariableOp#^model/layer2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2T
(model/Final_layer/BiasAdd/ReadVariableOp(model/Final_layer/BiasAdd/ReadVariableOp2R
'model/Final_layer/MatMul/ReadVariableOp'model/Final_layer/MatMul/ReadVariableOp2J
#model/layer1/BiasAdd/ReadVariableOp#model/layer1/BiasAdd/ReadVariableOp2H
"model/layer1/MatMul/ReadVariableOp"model/layer1/MatMul/ReadVariableOp2J
#model/layer2/BiasAdd/ReadVariableOp#model/layer2/BiasAdd/ReadVariableOp2H
"model/layer2/MatMul/ReadVariableOp"model/layer2/MatMul/ReadVariableOp:% !

_user_specified_nameInput
�	
�
'__inference_model_layer_call_fn_1147814	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_11478052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameInput
�V
�

#__inference__traced_restore_1148129
file_prefix"
assignvariableop_layer1_kernel"
assignvariableop_1_layer1_bias$
 assignvariableop_2_layer2_kernel"
assignvariableop_3_layer2_bias)
%assignvariableop_4_final_layer_kernel'
#assignvariableop_5_final_layer_bias
assignvariableop_6_total
assignvariableop_7_count&
"assignvariableop_8_layer1_kernel_m$
 assignvariableop_9_layer1_bias_m'
#assignvariableop_10_layer2_kernel_m%
!assignvariableop_11_layer2_bias_m,
(assignvariableop_12_final_layer_kernel_m*
&assignvariableop_13_final_layer_bias_m'
#assignvariableop_14_layer1_kernel_v%
!assignvariableop_15_layer1_bias_v'
#assignvariableop_16_layer2_kernel_v%
!assignvariableop_17_layer2_bias_v,
(assignvariableop_18_final_layer_kernel_v*
&assignvariableop_19_final_layer_bias_v
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_final_layer_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_final_layer_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_layer1_kernel_mIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_layer1_bias_mIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_layer2_kernel_mIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_layer2_bias_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp(assignvariableop_12_final_layer_kernel_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_final_layer_bias_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_layer1_kernel_vIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_layer1_bias_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_layer2_kernel_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_layer2_bias_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_final_layer_kernel_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp&assignvariableop_19_final_layer_bias_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20�
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�
�
B__inference_model_layer_call_and_return_conditional_losses_1147898

inputs)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource.
*final_layer_matmul_readvariableop_resource/
+final_layer_biasadd_readvariableop_resource
identity��"Final_layer/BiasAdd/ReadVariableOp�!Final_layer/MatMul/ReadVariableOp�layer1/BiasAdd/ReadVariableOp�layer1/MatMul/ReadVariableOp�layer2/BiasAdd/ReadVariableOp�layer2/MatMul/ReadVariableOp�
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02
layer1/MatMul/ReadVariableOp�
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
layer1/MatMul�
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
layer1/BiasAdd/ReadVariableOp�
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
layer1/BiasAddm
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
layer1/Relu�
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02
layer2/MatMul/ReadVariableOp�
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
layer2/MatMul�
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
layer2/BiasAdd/ReadVariableOp�
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
layer2/Relu�
!Final_layer/MatMul/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02#
!Final_layer/MatMul/ReadVariableOp�
Final_layer/MatMulMatMullayer2/Relu:activations:0)Final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Final_layer/MatMul�
"Final_layer/BiasAdd/ReadVariableOpReadVariableOp+final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"Final_layer/BiasAdd/ReadVariableOp�
Final_layer/BiasAddBiasAddFinal_layer/MatMul:product:0*Final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Final_layer/BiasAdd�
IdentityIdentityFinal_layer/BiasAdd:output:0#^Final_layer/BiasAdd/ReadVariableOp"^Final_layer/MatMul/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2H
"Final_layer/BiasAdd/ReadVariableOp"Final_layer/BiasAdd/ReadVariableOp2F
!Final_layer/MatMul/ReadVariableOp!Final_layer/MatMul/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1147805

inputs)
%layer1_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_2)
%layer2_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_2.
*final_layer_statefulpartitionedcall_args_1.
*final_layer_statefulpartitionedcall_args_2
identity��#Final_layer/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinputs%layer1_statefulpartitionedcall_args_1%layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_11477182 
layer1/StatefulPartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0%layer2_statefulpartitionedcall_args_1%layer2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_11477412 
layer2/StatefulPartitionedCall�
#Final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0*final_layer_statefulpartitionedcall_args_1*final_layer_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_Final_layer_layer_call_and_return_conditional_losses_11477632%
#Final_layer/StatefulPartitionedCall�
IdentityIdentity,Final_layer/StatefulPartitionedCall:output:0$^Final_layer/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2J
#Final_layer/StatefulPartitionedCall#Final_layer/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_layer2_layer_call_fn_1147956

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_11477412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_layer1_layer_call_fn_1147938

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_11477182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1147789	
input)
%layer1_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_2)
%layer2_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_2.
*final_layer_statefulpartitionedcall_args_1.
*final_layer_statefulpartitionedcall_args_2
identity��#Final_layer/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinput%layer1_statefulpartitionedcall_args_1%layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_11477182 
layer1/StatefulPartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0%layer2_statefulpartitionedcall_args_1%layer2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_11477412 
layer2/StatefulPartitionedCall�
#Final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0*final_layer_statefulpartitionedcall_args_1*final_layer_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_Final_layer_layer_call_and_return_conditional_losses_11477632%
#Final_layer/StatefulPartitionedCall�
IdentityIdentity,Final_layer/StatefulPartitionedCall:output:0$^Final_layer/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2J
#Final_layer/StatefulPartitionedCall#Final_layer/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:% !

_user_specified_nameInput
�
�
B__inference_model_layer_call_and_return_conditional_losses_1147874

inputs)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource.
*final_layer_matmul_readvariableop_resource/
+final_layer_biasadd_readvariableop_resource
identity��"Final_layer/BiasAdd/ReadVariableOp�!Final_layer/MatMul/ReadVariableOp�layer1/BiasAdd/ReadVariableOp�layer1/MatMul/ReadVariableOp�layer2/BiasAdd/ReadVariableOp�layer2/MatMul/ReadVariableOp�
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02
layer1/MatMul/ReadVariableOp�
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
layer1/MatMul�
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
layer1/BiasAdd/ReadVariableOp�
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
layer1/BiasAddm
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
layer1/Relu�
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02
layer2/MatMul/ReadVariableOp�
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
layer2/MatMul�
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
layer2/BiasAdd/ReadVariableOp�
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������	2
layer2/Relu�
!Final_layer/MatMul/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02#
!Final_layer/MatMul/ReadVariableOp�
Final_layer/MatMulMatMullayer2/Relu:activations:0)Final_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Final_layer/MatMul�
"Final_layer/BiasAdd/ReadVariableOpReadVariableOp+final_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"Final_layer/BiasAdd/ReadVariableOp�
Final_layer/BiasAddBiasAddFinal_layer/MatMul:product:0*Final_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Final_layer/BiasAdd�
IdentityIdentityFinal_layer/BiasAdd:output:0#^Final_layer/BiasAdd/ReadVariableOp"^Final_layer/MatMul/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2H
"Final_layer/BiasAdd/ReadVariableOp"Final_layer/BiasAdd/ReadVariableOp2F
!Final_layer/MatMul/ReadVariableOp!Final_layer/MatMul/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
-__inference_Final_layer_layer_call_fn_1147973

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_Final_layer_layer_call_and_return_conditional_losses_11477632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1147776	
input)
%layer1_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_2)
%layer2_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_2.
*final_layer_statefulpartitionedcall_args_1.
*final_layer_statefulpartitionedcall_args_2
identity��#Final_layer/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinput%layer1_statefulpartitionedcall_args_1%layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_11477182 
layer1/StatefulPartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0%layer2_statefulpartitionedcall_args_1%layer2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_11477412 
layer2/StatefulPartitionedCall�
#Final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0*final_layer_statefulpartitionedcall_args_1*final_layer_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_Final_layer_layer_call_and_return_conditional_losses_11477632%
#Final_layer/StatefulPartitionedCall�
IdentityIdentity,Final_layer/StatefulPartitionedCall:output:0$^Final_layer/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2J
#Final_layer/StatefulPartitionedCall#Final_layer/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:% !

_user_specified_nameInput
�	
�
C__inference_layer1_layer_call_and_return_conditional_losses_1147718

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������	2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�1
�
 __inference__traced_save_1148057
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop1
-savev2_final_layer_kernel_read_readvariableop/
+savev2_final_layer_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop.
*savev2_layer1_kernel_m_read_readvariableop,
(savev2_layer1_bias_m_read_readvariableop.
*savev2_layer2_kernel_m_read_readvariableop,
(savev2_layer2_bias_m_read_readvariableop3
/savev2_final_layer_kernel_m_read_readvariableop1
-savev2_final_layer_bias_m_read_readvariableop.
*savev2_layer1_kernel_v_read_readvariableop,
(savev2_layer1_bias_v_read_readvariableop.
*savev2_layer2_kernel_v_read_readvariableop,
(savev2_layer2_bias_v_read_readvariableop3
/savev2_final_layer_kernel_v_read_readvariableop1
-savev2_final_layer_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_72a4edf6573a4b9386cc3f157e8be18f/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop-savev2_final_layer_kernel_read_readvariableop+savev2_final_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop*savev2_layer1_kernel_m_read_readvariableop(savev2_layer1_bias_m_read_readvariableop*savev2_layer2_kernel_m_read_readvariableop(savev2_layer2_bias_m_read_readvariableop/savev2_final_layer_kernel_m_read_readvariableop-savev2_final_layer_bias_m_read_readvariableop*savev2_layer1_kernel_v_read_readvariableop(savev2_layer1_bias_v_read_readvariableop*savev2_layer2_kernel_v_read_readvariableop(savev2_layer2_bias_v_read_readvariableop/savev2_final_layer_kernel_v_read_readvariableop-savev2_final_layer_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *"
dtypes
22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	:	:		:	:	:: : :	:	:		:	:	::	:	:		:	:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�	
�
'__inference_model_layer_call_fn_1147909

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_11478052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1147829

inputs)
%layer1_statefulpartitionedcall_args_1)
%layer1_statefulpartitionedcall_args_2)
%layer2_statefulpartitionedcall_args_1)
%layer2_statefulpartitionedcall_args_2.
*final_layer_statefulpartitionedcall_args_1.
*final_layer_statefulpartitionedcall_args_2
identity��#Final_layer/StatefulPartitionedCall�layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinputs%layer1_statefulpartitionedcall_args_1%layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer1_layer_call_and_return_conditional_losses_11477182 
layer1/StatefulPartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0%layer2_statefulpartitionedcall_args_1%layer2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������	*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layer2_layer_call_and_return_conditional_losses_11477412 
layer2/StatefulPartitionedCall�
#Final_layer/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0*final_layer_statefulpartitionedcall_args_1*final_layer_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_Final_layer_layer_call_and_return_conditional_losses_11477632%
#Final_layer/StatefulPartitionedCall�
IdentityIdentity,Final_layer/StatefulPartitionedCall:output:0$^Final_layer/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::2J
#Final_layer/StatefulPartitionedCall#Final_layer/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
H__inference_Final_layer_layer_call_and_return_conditional_losses_1147763

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
'__inference_model_layer_call_fn_1147920

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_11478292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1147850	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__wrapped_model_11477032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameInput"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
Input.
serving_default_Input:0���������?
Final_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�"
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
*E&call_and_return_all_conditional_losses
F_default_save_signature
G__call__"� 
_tf_keras_model�{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Final_layer", "inbound_nodes": [[["layer2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Final_layer", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Final_layer", "inbound_nodes": [[["layer2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Final_layer", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": ["mape"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.002, "decay": 0.004, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 2], "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Final_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}}
m9m:m;m<m=m>v?v@vAvBvCvD"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
layer_regularization_losses

layers
non_trainable_variables
trainable_variables
regularization_losses
 metrics
G__call__
F_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Nserving_default"
signature_map
:	2layer1/kernel
:	2layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
!layer_regularization_losses

"layers
#non_trainable_variables
trainable_variables
regularization_losses
$metrics
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
:		2layer2/kernel
:	2layer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
%layer_regularization_losses

&layers
'non_trainable_variables
trainable_variables
regularization_losses
(metrics
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
$:"	2Final_layer/kernel
:2Final_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
)layer_regularization_losses

*layers
+non_trainable_variables
trainable_variables
regularization_losses
,metrics
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
'
-0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	.total
	/count
0
_fn_kwargs
1	variables
2trainable_variables
3regularization_losses
4	keras_api
*O&call_and_return_all_conditional_losses
P__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "mape", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mape", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
1	variables
5layer_regularization_losses

6layers
7non_trainable_variables
2trainable_variables
3regularization_losses
8metrics
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
:	2layer1/kernel/m
:	2layer1/bias/m
:		2layer2/kernel/m
:	2layer2/bias/m
$:"	2Final_layer/kernel/m
:2Final_layer/bias/m
:	2layer1/kernel/v
:	2layer1/bias/v
:		2layer2/kernel/v
:	2layer2/bias/v
$:"	2Final_layer/kernel/v
:2Final_layer/bias/v
�2�
B__inference_model_layer_call_and_return_conditional_losses_1147776
B__inference_model_layer_call_and_return_conditional_losses_1147789
B__inference_model_layer_call_and_return_conditional_losses_1147898
B__inference_model_layer_call_and_return_conditional_losses_1147874�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_1147703�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *$�!
�
Input���������
�2�
'__inference_model_layer_call_fn_1147838
'__inference_model_layer_call_fn_1147909
'__inference_model_layer_call_fn_1147814
'__inference_model_layer_call_fn_1147920�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_layer1_layer_call_and_return_conditional_losses_1147931�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_layer1_layer_call_fn_1147938�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_layer2_layer_call_and_return_conditional_losses_1147949�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_layer2_layer_call_fn_1147956�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_Final_layer_layer_call_and_return_conditional_losses_1147966�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_Final_layer_layer_call_fn_1147973�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2B0
%__inference_signature_wrapper_1147850Input
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
H__inference_Final_layer_layer_call_and_return_conditional_losses_1147966\/�,
%�"
 �
inputs���������	
� "%�"
�
0���������
� �
-__inference_Final_layer_layer_call_fn_1147973O/�,
%�"
 �
inputs���������	
� "�����������
"__inference__wrapped_model_1147703s.�+
$�!
�
Input���������
� "9�6
4
Final_layer%�"
Final_layer����������
C__inference_layer1_layer_call_and_return_conditional_losses_1147931\/�,
%�"
 �
inputs���������
� "%�"
�
0���������	
� {
(__inference_layer1_layer_call_fn_1147938O/�,
%�"
 �
inputs���������
� "����������	�
C__inference_layer2_layer_call_and_return_conditional_losses_1147949\/�,
%�"
 �
inputs���������	
� "%�"
�
0���������	
� {
(__inference_layer2_layer_call_fn_1147956O/�,
%�"
 �
inputs���������	
� "����������	�
B__inference_model_layer_call_and_return_conditional_losses_1147776g6�3
,�)
�
Input���������
p

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1147789g6�3
,�)
�
Input���������
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1147874h7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_1147898h7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
'__inference_model_layer_call_fn_1147814Z6�3
,�)
�
Input���������
p

 
� "�����������
'__inference_model_layer_call_fn_1147838Z6�3
,�)
�
Input���������
p 

 
� "�����������
'__inference_model_layer_call_fn_1147909[7�4
-�*
 �
inputs���������
p

 
� "�����������
'__inference_model_layer_call_fn_1147920[7�4
-�*
 �
inputs���������
p 

 
� "�����������
%__inference_signature_wrapper_1147850|7�4
� 
-�*
(
Input�
Input���������"9�6
4
Final_layer%�"
Final_layer���������