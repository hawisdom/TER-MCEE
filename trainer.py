import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as prf
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from data_process import event_roles
from datasets import my_collate


torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_input_from_batch(batch):
    inputs = { 'token_ids':batch[0],
               'pos_ids':batch[1],
               'dep_ids':batch[2],
               'sen_pos_ids':batch[3],
               'word_pos_ids':batch[4],
               'parent_ids':batch[5],
               'ppos_ids':batch[6],
               'pdep_ids':batch[7],
               'event_ids':batch[8]
                }
    type_labels = batch[9]

    return inputs, type_labels


def get_collate_fn():
    return my_collate

def train(args,model,train_dataset,train_labels_weight,dev_dataset,dev_labels_weight,test_dataset,test_labels_weight):
    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn()
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,batch_size=args.train_batch_size,collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epoch = 0

    max_avg = 0
    max_avg_epoch = 0

    f = open('./output/result.txt','w',encoding='utf-8')

    for _ in train_iterator:
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs,types_labels = get_input_from_batch(batch)
            logits = model(**inputs)

            loss = 0
            for i,(type,roles) in enumerate(event_roles.items()):
                type_weight = torch.from_numpy(np.array(train_labels_weight[type],dtype=np.float32)).to(args.device)
                loss += F.cross_entropy(logits[i], types_labels[i], weight=type_weight)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("  train_loss: %s", str((tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

        torch.save(model, './output/model_' + str(epoch))
        avg_result,eval_loss = evaluate(args, dev_dataset, model, dev_labels_weight, f)
        tb_writer.add_scalar('eval_loss', eval_loss, global_step)

        if avg_result > max_avg:
            max_avg = avg_result
            max_avg_epoch = epoch
        epoch += 1

    model = torch.load('./output/model_' + str(max_avg_epoch))
    logger.info('***** Test results *****')
    evaluate(args, test_dataset, model, test_labels_weight, f)

    tb_writer.close()
    f.close()

def evaluate(args, eval_dataset, model,test_labels_weight,f):
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn()
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,collate_fn=collate_fn)
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    type_preds = {}
    out_type_label_ids = {}

    with torch.no_grad():
        for batch in eval_dataloader:
            model.eval()

            batch = tuple(t.to(args.device) for t in batch)
            inputs, type_labels = get_input_from_batch(batch)
            logits = model(**inputs)

            loss = 0
            for i, (type, roles) in enumerate(event_roles.items()):
                type_weight = torch.from_numpy(np.array(test_labels_weight[type], dtype=np.float32)).to(args.device)
                loss += F.cross_entropy(logits[i], type_labels[i], weight=type_weight)

            eval_loss += loss.mean().item()
            nb_eval_steps += 1

            if len(type_preds) == 0:
                for etype_id,(type,roles) in enumerate(event_roles.items()):
                    type_preds[type] = []
                    out_type_label_ids[type] = []
                    type_preds[type] = logits[etype_id].detach().cpu().numpy()
                    out_type_label_ids[type] = type_labels[etype_id].detach().cpu().numpy()
            else:
                for etype_id, (type, roles) in enumerate(event_roles.items()):
                    type_preds[type] = np.append(type_preds[type], logits[etype_id].detach().cpu().numpy(), axis=0)
                    out_type_label_ids[type] = np.append(out_type_label_ids[type], type_labels[etype_id].detach().cpu().numpy(), axis=0)

    for etype_id, (type, roles) in enumerate(event_roles.items()):
        type_preds[type] = np.argmax(type_preds[type], axis=1)


    eval_loss = eval_loss / nb_eval_steps

    logger.info(" eval loss: %s", str(eval_loss))
    logger.info('***** Eval results *****')
    avg_result = 0
    for etype_id, (type, roles) in enumerate(event_roles.items()):
        type_result = prf_comput(type_preds[type], out_type_label_ids[type])
        logger.info("************%s*************",type)
        avg_result += type_result['f1']
        for key in type_result.keys():
            logger.info("  %s = %s", key, str(type_result[key]))
            f.write('ef' + key + '=' + str(type_result[key]) + '\n')

    return avg_result/len(event_roles),eval_loss

def prf_comput(preds,labels):
    role_preds = []
    role_labels = []
    for i, label in enumerate(labels):
        if label > 0:
            role_preds.append(preds[i])
            role_labels.append(labels[i])

    pre, recall, f1, _ = prf(y_true=role_labels, y_pred=role_preds, average='micro')
    return {
        "pre": pre,
        "recall": recall,
        "f1": f1
    }