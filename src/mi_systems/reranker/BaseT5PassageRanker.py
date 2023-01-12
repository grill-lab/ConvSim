from transformers import T5ForConditionalGeneration
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseT5PassageRanker(T5ForConditionalGeneration):

    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, decoder_past_key_value_states=None,
                use_cache=None, labels=None, inputs_embeds=None, decoder_inputs_embeds=None,
                head_mask=None, output_attentions=None, output_hidden_states=None, **kwargs):
      
      input_ids = input_ids.to(device)
      attn_mask = attention_mask.to(device)
      decode_ids = torch.full((input_ids.size(0), 1), self.config.decoder_start_token_id, dtype=torch.long).to(device)
      encoder_outputs = self.encoder(input_ids, attention_mask=attn_mask)
      next_token_logits = None

      for _ in range(1):
        model_inputs = self.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attn_mask,
            use_cache=True)
        
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=model_inputs['attention_mask'],
            use_cache=model_inputs['use_cache'],
            encoder_hidden_states=model_inputs['encoder_outputs'].last_hidden_state
        )
        next_token_logits = outputs[0][:, -1, :]  
        decode_ids = torch.cat([decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1)
      
      next_token_logits = next_token_logits[:, [54, 219]]
      next_token_logits = torch.nn.functional.log_softmax(next_token_logits, dim=1)
      batch_log_probs = next_token_logits[:, 1].tolist()

      return batch_log_probs