��
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
shapeshape�"serve*2.1.02v2.1.0-rc2-17-ge5bf8de8߼
x
layers1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayers1/kernel
q
"layers1/kernel/Read/ReadVariableOpReadVariableOplayers1/kernel*
_output_shapes

:*
dtype0
p
layers1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayers1/bias
i
 layers1/bias/Read/ReadVariableOpReadVariableOplayers1/bias*
_output_shapes
:*
dtype0
x
layers2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayers2/kernel
q
"layers2/kernel/Read/ReadVariableOpReadVariableOplayers2/kernel*
_output_shapes

:*
dtype0
p
layers2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayers2/bias
i
 layers2/bias/Read/ReadVariableOpReadVariableOplayers2/bias*
_output_shapes
:*
dtype0
x
layers3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayers3/kernel
q
"layers3/kernel/Read/ReadVariableOpReadVariableOplayers3/kernel*
_output_shapes

:*
dtype0
p
layers3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayers3/bias
i
 layers3/bias/Read/ReadVariableOpReadVariableOplayers3/bias*
_output_shapes
:*
dtype0
�
Final_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameFinal_layer/kernel
y
&Final_layer/kernel/Read/ReadVariableOpReadVariableOpFinal_layer/kernel*
_output_shapes

:*
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
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
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
�
Nadam/layers1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/layers1/kernel/m
�
*Nadam/layers1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/layers1/kernel/m*
_output_shapes

:*
dtype0
�
Nadam/layers1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/layers1/bias/m
y
(Nadam/layers1/bias/m/Read/ReadVariableOpReadVariableOpNadam/layers1/bias/m*
_output_shapes
:*
dtype0
�
Nadam/layers2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/layers2/kernel/m
�
*Nadam/layers2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/layers2/kernel/m*
_output_shapes

:*
dtype0
�
Nadam/layers2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/layers2/bias/m
y
(Nadam/layers2/bias/m/Read/ReadVariableOpReadVariableOpNadam/layers2/bias/m*
_output_shapes
:*
dtype0
�
Nadam/layers3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/layers3/kernel/m
�
*Nadam/layers3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/layers3/kernel/m*
_output_shapes

:*
dtype0
�
Nadam/layers3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/layers3/bias/m
y
(Nadam/layers3/bias/m/Read/ReadVariableOpReadVariableOpNadam/layers3/bias/m*
_output_shapes
:*
dtype0
�
Nadam/Final_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameNadam/Final_layer/kernel/m
�
.Nadam/Final_layer/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Final_layer/kernel/m*
_output_shapes

:*
dtype0
�
Nadam/Final_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameNadam/Final_layer/bias/m
�
,Nadam/Final_layer/bias/m/Read/ReadVariableOpReadVariableOpNadam/Final_layer/bias/m*
_output_shapes
:*
dtype0
�
Nadam/layers1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/layers1/kernel/v
�
*Nadam/layers1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/layers1/kernel/v*
_output_shapes

:*
dtype0
�
Nadam/layers1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/layers1/bias/v
y
(Nadam/layers1/bias/v/Read/ReadVariableOpReadVariableOpNadam/layers1/bias/v*
_output_shapes
:*
dtype0
�
Nadam/layers2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/layers2/kernel/v
�
*Nadam/layers2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/layers2/kernel/v*
_output_shapes

:*
dtype0
�
Nadam/layers2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/layers2/bias/v
y
(Nadam/layers2/bias/v/Read/ReadVariableOpReadVariableOpNadam/layers2/bias/v*
_output_shapes
:*
dtype0
�
Nadam/layers3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameNadam/layers3/kernel/v
�
*Nadam/layers3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/layers3/kernel/v*
_output_shapes

:*
dtype0
�
Nadam/layers3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameNadam/layers3/bias/v
y
(Nadam/layers3/bias/v/Read/ReadVariableOpReadVariableOpNadam/layers3/bias/v*
_output_shapes
:*
dtype0
�
Nadam/Final_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameNadam/Final_layer/kernel/v
�
.Nadam/Final_layer/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Final_layer/kernel/v*
_output_shapes

:*
dtype0
�
Nadam/Final_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameNadam/Final_layer/bias/v
�
,Nadam/Final_layer/bias/v/Read/ReadVariableOpReadVariableOpNadam/Final_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�-
value�-B�- B�-
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
�
$iter

%beta_1

&beta_2
	'decay
(learning_rate
)momentum_cachemJmKmLmMmNmOmPmQvRvSvTvUvVvWvXvY
 
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
�
*non_trainable_variables
regularization_losses
+metrics
trainable_variables
,layer_regularization_losses

-layers
		variables
 
ZX
VARIABLE_VALUElayers1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayers1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
.non_trainable_variables
regularization_losses
/metrics
trainable_variables
0layer_regularization_losses

1layers
	variables
ZX
VARIABLE_VALUElayers2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayers2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
2non_trainable_variables
regularization_losses
3metrics
trainable_variables
4layer_regularization_losses

5layers
	variables
ZX
VARIABLE_VALUElayers3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayers3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
6non_trainable_variables
regularization_losses
7metrics
trainable_variables
8layer_regularization_losses

9layers
	variables
^\
VARIABLE_VALUEFinal_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEFinal_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
:non_trainable_variables
 regularization_losses
;metrics
!trainable_variables
<layer_regularization_losses

=layers
"	variables
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 

>0
 
#
0
1
2
3
4
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
 
 
 
 
x
	?total
	@count
A
_fn_kwargs
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
@1
�
Fnon_trainable_variables
Bregularization_losses
Gmetrics
Ctrainable_variables
Hlayer_regularization_losses

Ilayers
D	variables

?0
@1
 
 
 
~|
VARIABLE_VALUENadam/layers1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/layers1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/layers2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/layers2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/layers3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/layers3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/Final_layer/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/Final_layer/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/layers1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/layers1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/layers2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/layers2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/layers3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/layers3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/Final_layer/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/Final_layer/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_InputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputlayers1/kernellayers1/biaslayers2/kernellayers2/biaslayers3/kernellayers3/biasFinal_layer/kernelFinal_layer/bias*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_771654
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"layers1/kernel/Read/ReadVariableOp layers1/bias/Read/ReadVariableOp"layers2/kernel/Read/ReadVariableOp layers2/bias/Read/ReadVariableOp"layers3/kernel/Read/ReadVariableOp layers3/bias/Read/ReadVariableOp&Final_layer/kernel/Read/ReadVariableOp$Final_layer/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Nadam/layers1/kernel/m/Read/ReadVariableOp(Nadam/layers1/bias/m/Read/ReadVariableOp*Nadam/layers2/kernel/m/Read/ReadVariableOp(Nadam/layers2/bias/m/Read/ReadVariableOp*Nadam/layers3/kernel/m/Read/ReadVariableOp(Nadam/layers3/bias/m/Read/ReadVariableOp.Nadam/Final_layer/kernel/m/Read/ReadVariableOp,Nadam/Final_layer/bias/m/Read/ReadVariableOp*Nadam/layers1/kernel/v/Read/ReadVariableOp(Nadam/layers1/bias/v/Read/ReadVariableOp*Nadam/layers2/kernel/v/Read/ReadVariableOp(Nadam/layers2/bias/v/Read/ReadVariableOp*Nadam/layers3/kernel/v/Read/ReadVariableOp(Nadam/layers3/bias/v/Read/ReadVariableOp.Nadam/Final_layer/kernel/v/Read/ReadVariableOp,Nadam/Final_layer/bias/v/Read/ReadVariableOpConst*-
Tin&
$2"	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_771933
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayers1/kernellayers1/biaslayers2/kernellayers2/biaslayers3/kernellayers3/biasFinal_layer/kernelFinal_layer/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/layers1/kernel/mNadam/layers1/bias/mNadam/layers2/kernel/mNadam/layers2/bias/mNadam/layers3/kernel/mNadam/layers3/bias/mNadam/Final_layer/kernel/mNadam/Final_layer/bias/mNadam/layers1/kernel/vNadam/layers1/bias/vNadam/layers2/kernel/vNadam/layers2/bias/vNadam/layers3/kernel/vNadam/layers3/bias/vNadam/Final_layer/kernel/vNadam/Final_layer/bias/v*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_772041��
�	
�
C__inference_layers2_layer_call_and_return_conditional_losses_771771

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_771573	
input*
&layers1_statefulpartitionedcall_args_1*
&layers1_statefulpartitionedcall_args_2*
&layers2_statefulpartitionedcall_args_1*
&layers2_statefulpartitionedcall_args_2*
&layers3_statefulpartitionedcall_args_1*
&layers3_statefulpartitionedcall_args_2.
*final_layer_statefulpartitionedcall_args_1.
*final_layer_statefulpartitionedcall_args_2
identity��#Final_layer/StatefulPartitionedCall�layers1/StatefulPartitionedCall�layers2/StatefulPartitionedCall�layers3/StatefulPartitionedCall�
layers1/StatefulPartitionedCallStatefulPartitionedCallinput&layers1_statefulpartitionedcall_args_1&layers1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers1_layer_call_and_return_conditional_losses_7714762!
layers1/StatefulPartitionedCall�
layers2/StatefulPartitionedCallStatefulPartitionedCall(layers1/StatefulPartitionedCall:output:0&layers2_statefulpartitionedcall_args_1&layers2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers2_layer_call_and_return_conditional_losses_7714992!
layers2/StatefulPartitionedCall�
layers3/StatefulPartitionedCallStatefulPartitionedCall(layers2/StatefulPartitionedCall:output:0&layers3_statefulpartitionedcall_args_1&layers3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers3_layer_call_and_return_conditional_losses_7715222!
layers3/StatefulPartitionedCall�
#Final_layer/StatefulPartitionedCallStatefulPartitionedCall(layers3/StatefulPartitionedCall:output:0*final_layer_statefulpartitionedcall_args_1*final_layer_statefulpartitionedcall_args_2*
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
GPU2*0J 8*P
fKRI
G__inference_Final_layer_layer_call_and_return_conditional_losses_7715442%
#Final_layer/StatefulPartitionedCall�
IdentityIdentity,Final_layer/StatefulPartitionedCall:output:0$^Final_layer/StatefulPartitionedCall ^layers1/StatefulPartitionedCall ^layers2/StatefulPartitionedCall ^layers3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2J
#Final_layer/StatefulPartitionedCall#Final_layer/StatefulPartitionedCall2B
layers1/StatefulPartitionedCalllayers1/StatefulPartitionedCall2B
layers2/StatefulPartitionedCalllayers2/StatefulPartitionedCall2B
layers3/StatefulPartitionedCalllayers3/StatefulPartitionedCall:% !

_user_specified_nameInput
��
�
"__inference__traced_restore_772041
file_prefix#
assignvariableop_layers1_kernel#
assignvariableop_1_layers1_bias%
!assignvariableop_2_layers2_kernel#
assignvariableop_3_layers2_bias%
!assignvariableop_4_layers3_kernel#
assignvariableop_5_layers3_bias)
%assignvariableop_6_final_layer_kernel'
#assignvariableop_7_final_layer_bias!
assignvariableop_8_nadam_iter#
assignvariableop_9_nadam_beta_1$
 assignvariableop_10_nadam_beta_2#
assignvariableop_11_nadam_decay+
'assignvariableop_12_nadam_learning_rate,
(assignvariableop_13_nadam_momentum_cache
assignvariableop_14_total
assignvariableop_15_count.
*assignvariableop_16_nadam_layers1_kernel_m,
(assignvariableop_17_nadam_layers1_bias_m.
*assignvariableop_18_nadam_layers2_kernel_m,
(assignvariableop_19_nadam_layers2_bias_m.
*assignvariableop_20_nadam_layers3_kernel_m,
(assignvariableop_21_nadam_layers3_bias_m2
.assignvariableop_22_nadam_final_layer_kernel_m0
,assignvariableop_23_nadam_final_layer_bias_m.
*assignvariableop_24_nadam_layers1_kernel_v,
(assignvariableop_25_nadam_layers1_bias_v.
*assignvariableop_26_nadam_layers2_kernel_v,
(assignvariableop_27_nadam_layers2_bias_v.
*assignvariableop_28_nadam_layers3_kernel_v,
(assignvariableop_29_nadam_layers3_bias_v2
.assignvariableop_30_nadam_final_layer_kernel_v0
,assignvariableop_31_nadam_final_layer_bias_v
identity_33��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_layers1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_layers1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_layers2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_layers2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_layers3_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_layers3_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_final_layer_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_final_layer_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_nadam_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_nadam_beta_1Identity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp assignvariableop_10_nadam_beta_2Identity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_nadam_decayIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_nadam_learning_rateIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp(assignvariableop_13_nadam_momentum_cacheIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_nadam_layers1_kernel_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_nadam_layers1_bias_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_nadam_layers2_kernel_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_nadam_layers2_bias_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_nadam_layers3_kernel_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_nadam_layers3_bias_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp.assignvariableop_22_nadam_final_layer_kernel_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp,assignvariableop_23_nadam_final_layer_bias_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_nadam_layers1_kernel_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_nadam_layers1_bias_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_nadam_layers2_kernel_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_nadam_layers2_bias_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_nadam_layers3_kernel_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_nadam_layers3_bias_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp.assignvariableop_30_nadam_final_layer_kernel_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp,assignvariableop_31_nadam_final_layer_bias_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31�
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
NoOp�
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32�
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
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

�
&__inference_model_layer_call_fn_771632	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_7716212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameInput
�
�
(__inference_layers3_layer_call_fn_771796

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
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers3_layer_call_and_return_conditional_losses_7715222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
G__inference_Final_layer_layer_call_and_return_conditional_losses_771544

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�*
�
!__inference__wrapped_model_771461	
input0
,model_layers1_matmul_readvariableop_resource1
-model_layers1_biasadd_readvariableop_resource0
,model_layers2_matmul_readvariableop_resource1
-model_layers2_biasadd_readvariableop_resource0
,model_layers3_matmul_readvariableop_resource1
-model_layers3_biasadd_readvariableop_resource4
0model_final_layer_matmul_readvariableop_resource5
1model_final_layer_biasadd_readvariableop_resource
identity��(model/Final_layer/BiasAdd/ReadVariableOp�'model/Final_layer/MatMul/ReadVariableOp�$model/layers1/BiasAdd/ReadVariableOp�#model/layers1/MatMul/ReadVariableOp�$model/layers2/BiasAdd/ReadVariableOp�#model/layers2/MatMul/ReadVariableOp�$model/layers3/BiasAdd/ReadVariableOp�#model/layers3/MatMul/ReadVariableOp�
#model/layers1/MatMul/ReadVariableOpReadVariableOp,model_layers1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/layers1/MatMul/ReadVariableOp�
model/layers1/MatMulMatMulinput+model/layers1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/layers1/MatMul�
$model/layers1/BiasAdd/ReadVariableOpReadVariableOp-model_layers1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/layers1/BiasAdd/ReadVariableOp�
model/layers1/BiasAddBiasAddmodel/layers1/MatMul:product:0,model/layers1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/layers1/BiasAdd�
model/layers1/ReluRelumodel/layers1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/layers1/Relu�
#model/layers2/MatMul/ReadVariableOpReadVariableOp,model_layers2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/layers2/MatMul/ReadVariableOp�
model/layers2/MatMulMatMul model/layers1/Relu:activations:0+model/layers2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/layers2/MatMul�
$model/layers2/BiasAdd/ReadVariableOpReadVariableOp-model_layers2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/layers2/BiasAdd/ReadVariableOp�
model/layers2/BiasAddBiasAddmodel/layers2/MatMul:product:0,model/layers2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/layers2/BiasAdd�
model/layers2/ReluRelumodel/layers2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/layers2/Relu�
#model/layers3/MatMul/ReadVariableOpReadVariableOp,model_layers3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/layers3/MatMul/ReadVariableOp�
model/layers3/MatMulMatMul model/layers2/Relu:activations:0+model/layers3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/layers3/MatMul�
$model/layers3/BiasAdd/ReadVariableOpReadVariableOp-model_layers3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/layers3/BiasAdd/ReadVariableOp�
model/layers3/BiasAddBiasAddmodel/layers3/MatMul:product:0,model/layers3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/layers3/BiasAdd�
model/layers3/ReluRelumodel/layers3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/layers3/Relu�
'model/Final_layer/MatMul/ReadVariableOpReadVariableOp0model_final_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'model/Final_layer/MatMul/ReadVariableOp�
model/Final_layer/MatMulMatMul model/layers3/Relu:activations:0/model/Final_layer/MatMul/ReadVariableOp:value:0*
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
model/Final_layer/BiasAdd�
IdentityIdentity"model/Final_layer/BiasAdd:output:0)^model/Final_layer/BiasAdd/ReadVariableOp(^model/Final_layer/MatMul/ReadVariableOp%^model/layers1/BiasAdd/ReadVariableOp$^model/layers1/MatMul/ReadVariableOp%^model/layers2/BiasAdd/ReadVariableOp$^model/layers2/MatMul/ReadVariableOp%^model/layers3/BiasAdd/ReadVariableOp$^model/layers3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2T
(model/Final_layer/BiasAdd/ReadVariableOp(model/Final_layer/BiasAdd/ReadVariableOp2R
'model/Final_layer/MatMul/ReadVariableOp'model/Final_layer/MatMul/ReadVariableOp2L
$model/layers1/BiasAdd/ReadVariableOp$model/layers1/BiasAdd/ReadVariableOp2J
#model/layers1/MatMul/ReadVariableOp#model/layers1/MatMul/ReadVariableOp2L
$model/layers2/BiasAdd/ReadVariableOp$model/layers2/BiasAdd/ReadVariableOp2J
#model/layers2/MatMul/ReadVariableOp#model/layers2/MatMul/ReadVariableOp2L
$model/layers3/BiasAdd/ReadVariableOp$model/layers3/BiasAdd/ReadVariableOp2J
#model/layers3/MatMul/ReadVariableOp#model/layers3/MatMul/ReadVariableOp:% !

_user_specified_nameInput
�	
�
C__inference_layers3_layer_call_and_return_conditional_losses_771522

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
C__inference_layers3_layer_call_and_return_conditional_losses_771789

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�%
�
A__inference_model_layer_call_and_return_conditional_losses_771685

inputs*
&layers1_matmul_readvariableop_resource+
'layers1_biasadd_readvariableop_resource*
&layers2_matmul_readvariableop_resource+
'layers2_biasadd_readvariableop_resource*
&layers3_matmul_readvariableop_resource+
'layers3_biasadd_readvariableop_resource.
*final_layer_matmul_readvariableop_resource/
+final_layer_biasadd_readvariableop_resource
identity��"Final_layer/BiasAdd/ReadVariableOp�!Final_layer/MatMul/ReadVariableOp�layers1/BiasAdd/ReadVariableOp�layers1/MatMul/ReadVariableOp�layers2/BiasAdd/ReadVariableOp�layers2/MatMul/ReadVariableOp�layers3/BiasAdd/ReadVariableOp�layers3/MatMul/ReadVariableOp�
layers1/MatMul/ReadVariableOpReadVariableOp&layers1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layers1/MatMul/ReadVariableOp�
layers1/MatMulMatMulinputs%layers1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers1/MatMul�
layers1/BiasAdd/ReadVariableOpReadVariableOp'layers1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layers1/BiasAdd/ReadVariableOp�
layers1/BiasAddBiasAddlayers1/MatMul:product:0&layers1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers1/BiasAddp
layers1/ReluRelulayers1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layers1/Relu�
layers2/MatMul/ReadVariableOpReadVariableOp&layers2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layers2/MatMul/ReadVariableOp�
layers2/MatMulMatMullayers1/Relu:activations:0%layers2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers2/MatMul�
layers2/BiasAdd/ReadVariableOpReadVariableOp'layers2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layers2/BiasAdd/ReadVariableOp�
layers2/BiasAddBiasAddlayers2/MatMul:product:0&layers2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers2/BiasAddp
layers2/ReluRelulayers2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layers2/Relu�
layers3/MatMul/ReadVariableOpReadVariableOp&layers3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layers3/MatMul/ReadVariableOp�
layers3/MatMulMatMullayers2/Relu:activations:0%layers3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers3/MatMul�
layers3/BiasAdd/ReadVariableOpReadVariableOp'layers3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layers3/BiasAdd/ReadVariableOp�
layers3/BiasAddBiasAddlayers3/MatMul:product:0&layers3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers3/BiasAddp
layers3/ReluRelulayers3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layers3/Relu�
!Final_layer/MatMul/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!Final_layer/MatMul/ReadVariableOp�
Final_layer/MatMulMatMullayers3/Relu:activations:0)Final_layer/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityFinal_layer/BiasAdd:output:0#^Final_layer/BiasAdd/ReadVariableOp"^Final_layer/MatMul/ReadVariableOp^layers1/BiasAdd/ReadVariableOp^layers1/MatMul/ReadVariableOp^layers2/BiasAdd/ReadVariableOp^layers2/MatMul/ReadVariableOp^layers3/BiasAdd/ReadVariableOp^layers3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2H
"Final_layer/BiasAdd/ReadVariableOp"Final_layer/BiasAdd/ReadVariableOp2F
!Final_layer/MatMul/ReadVariableOp!Final_layer/MatMul/ReadVariableOp2@
layers1/BiasAdd/ReadVariableOplayers1/BiasAdd/ReadVariableOp2>
layers1/MatMul/ReadVariableOplayers1/MatMul/ReadVariableOp2@
layers2/BiasAdd/ReadVariableOplayers2/BiasAdd/ReadVariableOp2>
layers2/MatMul/ReadVariableOplayers2/MatMul/ReadVariableOp2@
layers3/BiasAdd/ReadVariableOplayers3/BiasAdd/ReadVariableOp2>
layers3/MatMul/ReadVariableOplayers3/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
,__inference_Final_layer_layer_call_fn_771813

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
GPU2*0J 8*P
fKRI
G__inference_Final_layer_layer_call_and_return_conditional_losses_7715442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_771557	
input*
&layers1_statefulpartitionedcall_args_1*
&layers1_statefulpartitionedcall_args_2*
&layers2_statefulpartitionedcall_args_1*
&layers2_statefulpartitionedcall_args_2*
&layers3_statefulpartitionedcall_args_1*
&layers3_statefulpartitionedcall_args_2.
*final_layer_statefulpartitionedcall_args_1.
*final_layer_statefulpartitionedcall_args_2
identity��#Final_layer/StatefulPartitionedCall�layers1/StatefulPartitionedCall�layers2/StatefulPartitionedCall�layers3/StatefulPartitionedCall�
layers1/StatefulPartitionedCallStatefulPartitionedCallinput&layers1_statefulpartitionedcall_args_1&layers1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers1_layer_call_and_return_conditional_losses_7714762!
layers1/StatefulPartitionedCall�
layers2/StatefulPartitionedCallStatefulPartitionedCall(layers1/StatefulPartitionedCall:output:0&layers2_statefulpartitionedcall_args_1&layers2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers2_layer_call_and_return_conditional_losses_7714992!
layers2/StatefulPartitionedCall�
layers3/StatefulPartitionedCallStatefulPartitionedCall(layers2/StatefulPartitionedCall:output:0&layers3_statefulpartitionedcall_args_1&layers3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers3_layer_call_and_return_conditional_losses_7715222!
layers3/StatefulPartitionedCall�
#Final_layer/StatefulPartitionedCallStatefulPartitionedCall(layers3/StatefulPartitionedCall:output:0*final_layer_statefulpartitionedcall_args_1*final_layer_statefulpartitionedcall_args_2*
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
GPU2*0J 8*P
fKRI
G__inference_Final_layer_layer_call_and_return_conditional_losses_7715442%
#Final_layer/StatefulPartitionedCall�
IdentityIdentity,Final_layer/StatefulPartitionedCall:output:0$^Final_layer/StatefulPartitionedCall ^layers1/StatefulPartitionedCall ^layers2/StatefulPartitionedCall ^layers3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2J
#Final_layer/StatefulPartitionedCall#Final_layer/StatefulPartitionedCall2B
layers1/StatefulPartitionedCalllayers1/StatefulPartitionedCall2B
layers2/StatefulPartitionedCalllayers2/StatefulPartitionedCall2B
layers3/StatefulPartitionedCalllayers3/StatefulPartitionedCall:% !

_user_specified_nameInput
�

�
&__inference_model_layer_call_fn_771729

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_7715922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�

�
&__inference_model_layer_call_fn_771603	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_7715922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameInput
�

�
&__inference_model_layer_call_fn_771742

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_7716212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
G__inference_Final_layer_layer_call_and_return_conditional_losses_771806

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�B
�
__inference__traced_save_771933
file_prefix-
)savev2_layers1_kernel_read_readvariableop+
'savev2_layers1_bias_read_readvariableop-
)savev2_layers2_kernel_read_readvariableop+
'savev2_layers2_bias_read_readvariableop-
)savev2_layers3_kernel_read_readvariableop+
'savev2_layers3_bias_read_readvariableop1
-savev2_final_layer_kernel_read_readvariableop/
+savev2_final_layer_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_nadam_layers1_kernel_m_read_readvariableop3
/savev2_nadam_layers1_bias_m_read_readvariableop5
1savev2_nadam_layers2_kernel_m_read_readvariableop3
/savev2_nadam_layers2_bias_m_read_readvariableop5
1savev2_nadam_layers3_kernel_m_read_readvariableop3
/savev2_nadam_layers3_bias_m_read_readvariableop9
5savev2_nadam_final_layer_kernel_m_read_readvariableop7
3savev2_nadam_final_layer_bias_m_read_readvariableop5
1savev2_nadam_layers1_kernel_v_read_readvariableop3
/savev2_nadam_layers1_bias_v_read_readvariableop5
1savev2_nadam_layers2_kernel_v_read_readvariableop3
/savev2_nadam_layers2_bias_v_read_readvariableop5
1savev2_nadam_layers3_kernel_v_read_readvariableop3
/savev2_nadam_layers3_bias_v_read_readvariableop9
5savev2_nadam_final_layer_kernel_v_read_readvariableop7
3savev2_nadam_final_layer_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ab49d38f2a134f81a5981a69a6eb183a/part2
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layers1_kernel_read_readvariableop'savev2_layers1_bias_read_readvariableop)savev2_layers2_kernel_read_readvariableop'savev2_layers2_bias_read_readvariableop)savev2_layers3_kernel_read_readvariableop'savev2_layers3_bias_read_readvariableop-savev2_final_layer_kernel_read_readvariableop+savev2_final_layer_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_nadam_layers1_kernel_m_read_readvariableop/savev2_nadam_layers1_bias_m_read_readvariableop1savev2_nadam_layers2_kernel_m_read_readvariableop/savev2_nadam_layers2_bias_m_read_readvariableop1savev2_nadam_layers3_kernel_m_read_readvariableop/savev2_nadam_layers3_bias_m_read_readvariableop5savev2_nadam_final_layer_kernel_m_read_readvariableop3savev2_nadam_final_layer_bias_m_read_readvariableop1savev2_nadam_layers1_kernel_v_read_readvariableop/savev2_nadam_layers1_bias_v_read_readvariableop1savev2_nadam_layers2_kernel_v_read_readvariableop/savev2_nadam_layers2_bias_v_read_readvariableop1savev2_nadam_layers3_kernel_v_read_readvariableop/savev2_nadam_layers3_bias_v_read_readvariableop5savev2_nadam_final_layer_kernel_v_read_readvariableop3savev2_nadam_final_layer_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
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
�: ::::::::: : : : : : : : ::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�	
�
C__inference_layers2_layer_call_and_return_conditional_losses_771499

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
C__inference_layers1_layer_call_and_return_conditional_losses_771476

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_layers1_layer_call_fn_771760

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
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers1_layer_call_and_return_conditional_losses_7714762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_771592

inputs*
&layers1_statefulpartitionedcall_args_1*
&layers1_statefulpartitionedcall_args_2*
&layers2_statefulpartitionedcall_args_1*
&layers2_statefulpartitionedcall_args_2*
&layers3_statefulpartitionedcall_args_1*
&layers3_statefulpartitionedcall_args_2.
*final_layer_statefulpartitionedcall_args_1.
*final_layer_statefulpartitionedcall_args_2
identity��#Final_layer/StatefulPartitionedCall�layers1/StatefulPartitionedCall�layers2/StatefulPartitionedCall�layers3/StatefulPartitionedCall�
layers1/StatefulPartitionedCallStatefulPartitionedCallinputs&layers1_statefulpartitionedcall_args_1&layers1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers1_layer_call_and_return_conditional_losses_7714762!
layers1/StatefulPartitionedCall�
layers2/StatefulPartitionedCallStatefulPartitionedCall(layers1/StatefulPartitionedCall:output:0&layers2_statefulpartitionedcall_args_1&layers2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers2_layer_call_and_return_conditional_losses_7714992!
layers2/StatefulPartitionedCall�
layers3/StatefulPartitionedCallStatefulPartitionedCall(layers2/StatefulPartitionedCall:output:0&layers3_statefulpartitionedcall_args_1&layers3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers3_layer_call_and_return_conditional_losses_7715222!
layers3/StatefulPartitionedCall�
#Final_layer/StatefulPartitionedCallStatefulPartitionedCall(layers3/StatefulPartitionedCall:output:0*final_layer_statefulpartitionedcall_args_1*final_layer_statefulpartitionedcall_args_2*
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
GPU2*0J 8*P
fKRI
G__inference_Final_layer_layer_call_and_return_conditional_losses_7715442%
#Final_layer/StatefulPartitionedCall�
IdentityIdentity,Final_layer/StatefulPartitionedCall:output:0$^Final_layer/StatefulPartitionedCall ^layers1/StatefulPartitionedCall ^layers2/StatefulPartitionedCall ^layers3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2J
#Final_layer/StatefulPartitionedCall#Final_layer/StatefulPartitionedCall2B
layers1/StatefulPartitionedCalllayers1/StatefulPartitionedCall2B
layers2/StatefulPartitionedCalllayers2/StatefulPartitionedCall2B
layers3/StatefulPartitionedCalllayers3/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�%
�
A__inference_model_layer_call_and_return_conditional_losses_771716

inputs*
&layers1_matmul_readvariableop_resource+
'layers1_biasadd_readvariableop_resource*
&layers2_matmul_readvariableop_resource+
'layers2_biasadd_readvariableop_resource*
&layers3_matmul_readvariableop_resource+
'layers3_biasadd_readvariableop_resource.
*final_layer_matmul_readvariableop_resource/
+final_layer_biasadd_readvariableop_resource
identity��"Final_layer/BiasAdd/ReadVariableOp�!Final_layer/MatMul/ReadVariableOp�layers1/BiasAdd/ReadVariableOp�layers1/MatMul/ReadVariableOp�layers2/BiasAdd/ReadVariableOp�layers2/MatMul/ReadVariableOp�layers3/BiasAdd/ReadVariableOp�layers3/MatMul/ReadVariableOp�
layers1/MatMul/ReadVariableOpReadVariableOp&layers1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layers1/MatMul/ReadVariableOp�
layers1/MatMulMatMulinputs%layers1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers1/MatMul�
layers1/BiasAdd/ReadVariableOpReadVariableOp'layers1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layers1/BiasAdd/ReadVariableOp�
layers1/BiasAddBiasAddlayers1/MatMul:product:0&layers1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers1/BiasAddp
layers1/ReluRelulayers1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layers1/Relu�
layers2/MatMul/ReadVariableOpReadVariableOp&layers2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layers2/MatMul/ReadVariableOp�
layers2/MatMulMatMullayers1/Relu:activations:0%layers2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers2/MatMul�
layers2/BiasAdd/ReadVariableOpReadVariableOp'layers2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layers2/BiasAdd/ReadVariableOp�
layers2/BiasAddBiasAddlayers2/MatMul:product:0&layers2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers2/BiasAddp
layers2/ReluRelulayers2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layers2/Relu�
layers3/MatMul/ReadVariableOpReadVariableOp&layers3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layers3/MatMul/ReadVariableOp�
layers3/MatMulMatMullayers2/Relu:activations:0%layers3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers3/MatMul�
layers3/BiasAdd/ReadVariableOpReadVariableOp'layers3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layers3/BiasAdd/ReadVariableOp�
layers3/BiasAddBiasAddlayers3/MatMul:product:0&layers3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layers3/BiasAddp
layers3/ReluRelulayers3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layers3/Relu�
!Final_layer/MatMul/ReadVariableOpReadVariableOp*final_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!Final_layer/MatMul/ReadVariableOp�
Final_layer/MatMulMatMullayers3/Relu:activations:0)Final_layer/MatMul/ReadVariableOp:value:0*
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
IdentityIdentityFinal_layer/BiasAdd:output:0#^Final_layer/BiasAdd/ReadVariableOp"^Final_layer/MatMul/ReadVariableOp^layers1/BiasAdd/ReadVariableOp^layers1/MatMul/ReadVariableOp^layers2/BiasAdd/ReadVariableOp^layers2/MatMul/ReadVariableOp^layers3/BiasAdd/ReadVariableOp^layers3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2H
"Final_layer/BiasAdd/ReadVariableOp"Final_layer/BiasAdd/ReadVariableOp2F
!Final_layer/MatMul/ReadVariableOp!Final_layer/MatMul/ReadVariableOp2@
layers1/BiasAdd/ReadVariableOplayers1/BiasAdd/ReadVariableOp2>
layers1/MatMul/ReadVariableOplayers1/MatMul/ReadVariableOp2@
layers2/BiasAdd/ReadVariableOplayers2/BiasAdd/ReadVariableOp2>
layers2/MatMul/ReadVariableOplayers2/MatMul/ReadVariableOp2@
layers3/BiasAdd/ReadVariableOplayers3/BiasAdd/ReadVariableOp2>
layers3/MatMul/ReadVariableOplayers3/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_771621

inputs*
&layers1_statefulpartitionedcall_args_1*
&layers1_statefulpartitionedcall_args_2*
&layers2_statefulpartitionedcall_args_1*
&layers2_statefulpartitionedcall_args_2*
&layers3_statefulpartitionedcall_args_1*
&layers3_statefulpartitionedcall_args_2.
*final_layer_statefulpartitionedcall_args_1.
*final_layer_statefulpartitionedcall_args_2
identity��#Final_layer/StatefulPartitionedCall�layers1/StatefulPartitionedCall�layers2/StatefulPartitionedCall�layers3/StatefulPartitionedCall�
layers1/StatefulPartitionedCallStatefulPartitionedCallinputs&layers1_statefulpartitionedcall_args_1&layers1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers1_layer_call_and_return_conditional_losses_7714762!
layers1/StatefulPartitionedCall�
layers2/StatefulPartitionedCallStatefulPartitionedCall(layers1/StatefulPartitionedCall:output:0&layers2_statefulpartitionedcall_args_1&layers2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers2_layer_call_and_return_conditional_losses_7714992!
layers2/StatefulPartitionedCall�
layers3/StatefulPartitionedCallStatefulPartitionedCall(layers2/StatefulPartitionedCall:output:0&layers3_statefulpartitionedcall_args_1&layers3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers3_layer_call_and_return_conditional_losses_7715222!
layers3/StatefulPartitionedCall�
#Final_layer/StatefulPartitionedCallStatefulPartitionedCall(layers3/StatefulPartitionedCall:output:0*final_layer_statefulpartitionedcall_args_1*final_layer_statefulpartitionedcall_args_2*
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
GPU2*0J 8*P
fKRI
G__inference_Final_layer_layer_call_and_return_conditional_losses_7715442%
#Final_layer/StatefulPartitionedCall�
IdentityIdentity,Final_layer/StatefulPartitionedCall:output:0$^Final_layer/StatefulPartitionedCall ^layers1/StatefulPartitionedCall ^layers2/StatefulPartitionedCall ^layers3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2J
#Final_layer/StatefulPartitionedCall#Final_layer/StatefulPartitionedCall2B
layers1/StatefulPartitionedCalllayers1/StatefulPartitionedCall2B
layers2/StatefulPartitionedCalllayers2/StatefulPartitionedCall2B
layers3/StatefulPartitionedCalllayers3/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�

�
$__inference_signature_wrapper_771654	
input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_7714612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameInput
�	
�
C__inference_layers1_layer_call_and_return_conditional_losses_771753

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_layers2_layer_call_fn_771778

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
:���������*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_layers2_layer_call_and_return_conditional_losses_7714992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"�L
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�+
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
Z__call__
*[&call_and_return_all_conditional_losses
\_default_save_signature"�(
_tf_keras_model�({"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "layers1", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layers1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layers2", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layers2", "inbound_nodes": [[["layers1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layers3", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layers3", "inbound_nodes": [[["layers2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Final_layer", "inbound_nodes": [[["layers3", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Final_layer", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "layers1", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layers1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layers2", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layers2", "inbound_nodes": [[["layers1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layers3", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layers3", "inbound_nodes": [[["layers2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Final_layer", "inbound_nodes": [[["layers3", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Final_layer", 0, 0]]}}, "training_config": {"loss": "mse", "metrics": ["mape"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0020000000949949026, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 2], "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "layers1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layers1", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "layers2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layers2", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 21}}}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "layers3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layers3", "trainable": true, "dtype": "float32", "units": 21, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 21}}}}
�

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
c__call__
*d&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "Final_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Final_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 21}}}}
�
$iter

%beta_1

&beta_2
	'decay
(learning_rate
)momentum_cachemJmKmLmMmNmOmPmQvRvSvTvUvVvWvXvY"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
�
*non_trainable_variables
regularization_losses
+metrics
trainable_variables
,layer_regularization_losses

-layers
		variables
Z__call__
\_default_save_signature
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
,
eserving_default"
signature_map
 :2layers1/kernel
:2layers1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
.non_trainable_variables
regularization_losses
/metrics
trainable_variables
0layer_regularization_losses

1layers
	variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 :2layers2/kernel
:2layers2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
2non_trainable_variables
regularization_losses
3metrics
trainable_variables
4layer_regularization_losses

5layers
	variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 :2layers3/kernel
:2layers3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
6non_trainable_variables
regularization_losses
7metrics
trainable_variables
8layer_regularization_losses

9layers
	variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
$:"2Final_layer/kernel
:2Final_layer/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
:non_trainable_variables
 regularization_losses
;metrics
!trainable_variables
<layer_regularization_losses

=layers
"	variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	?total
	@count
A
_fn_kwargs
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
f__call__
*g&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "mape", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mape", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
�
Fnon_trainable_variables
Bregularization_losses
Gmetrics
Ctrainable_variables
Hlayer_regularization_losses

Ilayers
D	variables
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
&:$2Nadam/layers1/kernel/m
 :2Nadam/layers1/bias/m
&:$2Nadam/layers2/kernel/m
 :2Nadam/layers2/bias/m
&:$2Nadam/layers3/kernel/m
 :2Nadam/layers3/bias/m
*:(2Nadam/Final_layer/kernel/m
$:"2Nadam/Final_layer/bias/m
&:$2Nadam/layers1/kernel/v
 :2Nadam/layers1/bias/v
&:$2Nadam/layers2/kernel/v
 :2Nadam/layers2/bias/v
&:$2Nadam/layers3/kernel/v
 :2Nadam/layers3/bias/v
*:(2Nadam/Final_layer/kernel/v
$:"2Nadam/Final_layer/bias/v
�2�
&__inference_model_layer_call_fn_771729
&__inference_model_layer_call_fn_771632
&__inference_model_layer_call_fn_771742
&__inference_model_layer_call_fn_771603�
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
�2�
A__inference_model_layer_call_and_return_conditional_losses_771557
A__inference_model_layer_call_and_return_conditional_losses_771716
A__inference_model_layer_call_and_return_conditional_losses_771685
A__inference_model_layer_call_and_return_conditional_losses_771573�
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
!__inference__wrapped_model_771461�
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
�2�
(__inference_layers1_layer_call_fn_771760�
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
C__inference_layers1_layer_call_and_return_conditional_losses_771753�
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
(__inference_layers2_layer_call_fn_771778�
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
C__inference_layers2_layer_call_and_return_conditional_losses_771771�
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
(__inference_layers3_layer_call_fn_771796�
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
C__inference_layers3_layer_call_and_return_conditional_losses_771789�
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
,__inference_Final_layer_layer_call_fn_771813�
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
G__inference_Final_layer_layer_call_and_return_conditional_losses_771806�
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
1B/
$__inference_signature_wrapper_771654Input
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
G__inference_Final_layer_layer_call_and_return_conditional_losses_771806\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� 
,__inference_Final_layer_layer_call_fn_771813O/�,
%�"
 �
inputs���������
� "�����������
!__inference__wrapped_model_771461u.�+
$�!
�
Input���������
� "9�6
4
Final_layer%�"
Final_layer����������
C__inference_layers1_layer_call_and_return_conditional_losses_771753\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
(__inference_layers1_layer_call_fn_771760O/�,
%�"
 �
inputs���������
� "�����������
C__inference_layers2_layer_call_and_return_conditional_losses_771771\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
(__inference_layers2_layer_call_fn_771778O/�,
%�"
 �
inputs���������
� "�����������
C__inference_layers3_layer_call_and_return_conditional_losses_771789\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
(__inference_layers3_layer_call_fn_771796O/�,
%�"
 �
inputs���������
� "�����������
A__inference_model_layer_call_and_return_conditional_losses_771557i6�3
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
A__inference_model_layer_call_and_return_conditional_losses_771573i6�3
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
A__inference_model_layer_call_and_return_conditional_losses_771685j7�4
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
A__inference_model_layer_call_and_return_conditional_losses_771716j7�4
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
&__inference_model_layer_call_fn_771603\6�3
,�)
�
Input���������
p

 
� "�����������
&__inference_model_layer_call_fn_771632\6�3
,�)
�
Input���������
p 

 
� "�����������
&__inference_model_layer_call_fn_771729]7�4
-�*
 �
inputs���������
p

 
� "�����������
&__inference_model_layer_call_fn_771742]7�4
-�*
 �
inputs���������
p 

 
� "�����������
$__inference_signature_wrapper_771654~7�4
� 
-�*
(
Input�
Input���������"9�6
4
Final_layer%�"
Final_layer���������