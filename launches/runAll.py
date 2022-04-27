import os

cmd = ('set -e \n'+
'export PYTHONPATH=`pwd`:$PYTHONPATH \n'+
'WORK_DIR=$(pwd) \n'+
'SRC_DIR="${WORK_DIR}/src" \n'+
'python "${SRC_DIR}"/main.py train\\\n'+
'  --cuda=0 \\\n'+
'  --dataset=Altrasound \\\n'+
'  --model_type=%s \\\n'+
'  --kfold=%d \\\n'+
'  --epoch=%d \\\n'+
'  --lr=%f \\\n'+
'  --lr_epoch=%d \\\n'+
'  --batch_size=%d \\\n'+
'  --optimizer=%s \\\n'+
'  --num_workers=4 \\\n'+
'  --select_list=%s  \\\n'+
'  --show_log'
)

sgd_cmd = '  --sgd_m=%f \\\n  --sgd_w=%f \\\n'

# model_list = ['res50', 'efficientnet']
model_list = ['res50']
# kfold_list = [5,10,20]
kfold_list = [1]
lr_list = [0.0001]
epoch_list = [150]
lr_epoch_list = [40]
batch_size_list = [16]
# opt_list = ['sgd','adam']
opt_list = ['adam']
sgd_m_list = [0.9]
sgd_w_list = [5e-4]
select_lists = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

if __name__ == '__main__':
    for model in model_list:
        for kfold in kfold_list:
            for epoch in epoch_list:
                for lr in lr_list:
                    for lr_epoch in lr_epoch_list:
                        for batch_size in batch_size_list:
                            for select_list in select_lists:
                                for opt in opt_list:
                                    if opt=='sgd':
                                        for sgd_m in sgd_m_list:
                                            for sgd_w in sgd_w_list:
                                                print((cmd+sgd_cmd) % (model,kfold,epoch,lr,lr_epoch,batch_size,opt,select_list,sgd_m,sgd_w))
                                                # os.system((cmd+sgd_cmd) % (model,kfold,epoch,lr,lr_epoch,batch_size,opt,sgd_m,sgd_w))
                                    else:
                                        # print(cmd % (model,kfold,epoch,lr,lr_epoch,batch_size,opt))
                                        os.system(cmd % (model,kfold,epoch,lr,lr_epoch,batch_size,opt,select_list))
