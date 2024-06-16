import torch
from tokenizers import Tokenizer

# length penalty
def sequence_length_penalty(
    length: int,
    alpha: float=0.6
) -> float:
    return ((5 + length) / (5 + 1)) ** alpha

# beam search
def beam_search(
    model,
    config: dict,
    beam_size: int,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    src: str,
):
    model.eval()
    
    # special token id
    sos_token_id = tokenizer_src.token_to_id("<s>")
    eos_token_id = tokenizer_src.token_to_id("</s>")
    pad_token_id = tokenizer_src.token_to_id("<pad>")

    sos_token = torch.tensor([tokenizer_tgt.token_to_id("<s>")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("</s>")], dtype=torch.int64)
    
    device = config["device"]
    max_len = config["max_len"]

    enc_input_tokens = tokenizer_src.encode(src).ids
    src = torch.cat(
        [
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token,
        ],
        dim=0,
    ).to(device)

    src = torch.tensor(src, dtype=torch.int64).unsqueeze(0).to(device)
    src_attention_mask = (src != pad_token_id).type(torch.int64).to(device)

    encoder_output = model.get_encoder_out(
        input_ids=src,
        attention_mask=src_attention_mask
    ).last_hidden_state

    decoder_initial_input = torch.empty(1, 1).fill_(sos_token_id).type_as(src).to(device)
    
    candidates = [(decoder_initial_input, 0)]
    
    while True:
        if all([(cand[0][-1].item() == eos_token_id or cand.size(1) == max_len) for cand, _ in candidates]):
            break

        new_candidates = []

        for candidate, score in candidates:
            if candidate[0][-1].item() == eos_token_id or candidate.size(-1) == max_len:
                new_candidates.append((candidate, score))
                continue
            
            candidate_attention_mask = (candidate != pad_token_id).type_as(src_attention_mask).to(device)
            decoder_out = model.get_decoder_out(
                input_ids=candidate,
                attention_mask=candidate_attention_mask,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=src_attention_mask
            ).last_hidden_state
            
            out = model.out(decoder_out)
            prob = torch.nn.functional.log_softmax(out[:, -1], dim=1)
            prob = prob / sequence_length_penalty(candidate.size(-1), alpha=0.6)
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                new_candidate = torch.cat([candidate, token], dim=1)
                new_candidates.append((new_candidate, score + token_prob))

        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_size]

    pred_ids = candidates[0][0].squeeze()
    
    return pred_ids