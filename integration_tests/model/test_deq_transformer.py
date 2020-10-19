import torch

from deq_torch.models.transformers.deq_transformer import DEQTransformerLM


def test_deq_transformer():
    torch.set_default_tensor_type('torch.FloatTensor')
    model = DEQTransformerLM(n_token=500, n_layer=2,
                             eval_n_layer=24, n_head=12, d_model=120, d_head=10, d_inner=500,
                             dropout=0.1, dropatt=0.1, mem_len=100, tgt_len=100, tie_weights=True, d_embed=None)

    raw_data = torch.randint(0, 500, (200, 7)).long()
    data, target = raw_data[:75], raw_data[1:76]
    mems = None
    train_step=-1
    model.eval()

    model.train()
    ret = model(data, target, mems=mems, f_thres=50, b_thres=80, train_step=train_step)
    loss, mems = ret[0], ret[1:]
    loss = loss.float().mean().type_as(loss)
    loss.backward()
    print(model.func.dec_attn.qkv_net.weight.grad)