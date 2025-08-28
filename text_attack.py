import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
from bert_score.utils import get_idf_dict
import numpy as np
import time
import os
import argparse
import jiwer
import warnings
warnings.filterwarnings('ignore')


def wer(x, y):
    """Calculate Word Error Rate between two sequences"""
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])
    return jiwer.wer(x, y)


def bert_score(refs, cands, weights=None):
    """Calculate BERT score between reference and candidate embeddings"""
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # Remove first and last tokens
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R


def log_perplexity(logits, coeffs):
    """Calculate log perplexity"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


class BertAdversarialAttack:
    def __init__(self,
                 model_path='bert-base-uncased',
                 use_pretrained=True,
                 checkpoint_path=None,
                 device='cuda'):
        """
        Initialize the adversarial attack

        Args:
            model_path: Path to BERT model or model name
            use_pretrained: Whether to use pretrained model or load from checkpoint
            checkpoint_path: Path to model checkpoint if not using pretrained
            device: Device to run on
        """
        self.device = device
        self.model_path = model_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tokenizer.model_max_length = 512

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=4
        ).to(device)
        self.final_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=4
        ).to(device)

        if not use_pretrained and checkpoint_path:
            print(f'Loading checkpoint from {checkpoint_path}')
            self.model.load_state_dict(torch.load(checkpoint_path))
        self.final_model.load_state_dict(torch.load("bertmod/100.pth"))

        self.model.eval()
        self.final_model.eval()

        # Load reference model for perplexity calculation
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            'gpt2', output_hidden_states=True
        ).to(device)
        self.ref_model.eval()

        # Get embeddings
        with torch.no_grad():
            vocab_size = self.tokenizer.vocab_size
            self.embeddings = self.model.get_input_embeddings()(
                torch.arange(0, vocab_size).long().to(device)
            )
            self.ref_embeddings = self.ref_model.get_input_embeddings()(
                torch.arange(0, vocab_size).long().to(device)
            )

    def load_agnews_data(self, num_samples=100, start_idx=0):
        """Load AG News dataset"""
        dataset = load_dataset('ag_news')

        # Preprocess function for AG News
        def preprocess_function(examples):
            return self.tokenizer(
                examples['text'],
                max_length=256,
                truncation=True,
                padding='max_length'
            )

        # Encode dataset
        encoded_dataset = dataset.map(preprocess_function, batched=True)

        # Calculate IDF dictionary for BERTScore
        print("Computing IDF dictionary...")
        # Create a vocabulary mapping from token IDs
        idf_dict = {}
        vocab = self.tokenizer.get_vocab()

        # Sample texts for IDF calculation
        sample_texts = dataset['train']['text'][:5000]
        tokenized = self.tokenizer(sample_texts, truncation=True, padding=False)

        # Count document frequency for each token
        from collections import Counter
        doc_freq = Counter()
        for input_ids in tokenized['input_ids']:
            unique_tokens = set(input_ids)
            for token in unique_tokens:
                doc_freq[token] += 1

        # Calculate IDF
        total_docs = len(sample_texts)
        for token_id, freq in doc_freq.items():
            idf_dict[token_id] = np.log((total_docs + 1) / (freq + 1))

        return encoded_dataset, idf_dict

    def attack_sample(self,
                     input_ids,
                     label,
                     idf_dict=None,
                     num_iters=200,  # Increased for better convergence
                     lr=0.1,  # Reduced learning rate
                     batch_size=10,
                     lam_sim=5.0,  # Increased similarity weight
                     lam_perp=2.0,  # Increased perplexity weight
                     kappa=5.0,
                     initial_coeff=10,  # Reduced initial coefficient
                     constraint='bertscore_idf',
                     adv_loss='ce',
                     gumbel_samples=100,
                     embed_layer=-1,
                     only_word_substitution=True):  # New parameter
        """
        Attack a single sample

        Returns:
            Dictionary containing attack results
        """
        input_ids = torch.LongTensor(input_ids).to(self.device)

        # Get clean prediction
        with torch.no_grad():
            clean_logits = self.model(
                input_ids=input_ids.unsqueeze(0)
            ).logits.cpu()

        # Decode original text properly (skip special tokens)
        clean_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f'Original text: {clean_text}')
        print(f'Label: {label}, Clean prediction: {clean_logits.argmax().item()}')

        # Setup forbidden indices (CLS, SEP, and PAD tokens)
        forbidden = np.zeros(len(input_ids)).astype('bool')

        # Find special token positions
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id

        # Mark special tokens as forbidden
        for i, token_id in enumerate(input_ids.cpu().numpy()):
            if token_id in [cls_token_id, sep_token_id, pad_token_id]:
                forbidden[i] = True

        # If only_word_substitution, also forbid subword tokens (tokens starting with ##)
        if only_word_substitution:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy())
            for i, token in enumerate(tokens):
                if token.startswith('##'):
                    forbidden[i] = True

        forbidden_indices = torch.from_numpy(
            np.arange(0, len(input_ids))[forbidden]
        ).to(self.device)

        # Find the positions of actual content (between CLS and SEP, excluding PAD)
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            first_sep_pos = sep_positions[0].item()
            content_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            content_mask[1:first_sep_pos] = True  # Content is between CLS and first SEP
        else:
            content_mask = torch.ones_like(input_ids, dtype=torch.bool)
            content_mask[0] = False  # Skip CLS
            content_mask[input_ids == pad_token_id] = False  # Skip padding

        # Create vocabulary mask to restrict replacements
        vocab_mask = torch.ones(self.tokenizer.vocab_size, dtype=torch.bool)

        if only_word_substitution:
            # Only allow complete words (not subwords) as replacements
            for token_id, token in enumerate(self.tokenizer.get_vocab()):
                if token.startswith('##') or token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                    vocab_mask[token_id] = False

        # Initialize coefficients
        log_coeffs = torch.zeros(len(input_ids), self.embeddings.size(0))
        indices = torch.arange(log_coeffs.size(0)).long()
        log_coeffs[indices, input_ids.cpu()] = initial_coeff

        # Apply vocabulary mask - set forbidden tokens to very negative values
        if only_word_substitution:
            log_coeffs[:, ~vocab_mask] = -1e10

        log_coeffs = log_coeffs.to(self.device)
        log_coeffs.requires_grad = True

        # Get reference embeddings for similarity constraint
        with torch.no_grad():
            orig_output = self.ref_model(
                input_ids.unsqueeze(0)
            ).hidden_states[embed_layer]

            if constraint == 'bertscore_idf' and idf_dict:
                ref_weights = torch.FloatTensor(
                    [idf_dict.get(idx.item(), 1.0) for idx in input_ids]
                ).to(self.device)
                ref_weights /= ref_weights.sum()
            else:
                ref_weights = None

        optimizer = torch.optim.Adam([log_coeffs], lr=lr)

        # Attack loop
        adv_losses, ref_losses, perp_losses = [], [], []

        for i in range(num_iters):
            optimizer.zero_grad()

            # Sample from Gumbel-Softmax
            coeffs = F.gumbel_softmax(
                log_coeffs.unsqueeze(0).repeat(batch_size, 1, 1),
                hard=False
            )

            # Get input embeddings
            inputs_embeds = coeffs @ self.embeddings[None, :, :]

            # Forward pass
            pred = self.model(inputs_embeds=inputs_embeds).logits

            # Calculate adversarial loss
            if adv_loss == 'ce':
                adv_loss_val = -F.cross_entropy(
                    pred,
                    torch.ones(batch_size).long().to(self.device) * label
                )
            else:  # CW loss
                top_preds = pred.sort(descending=True)[1]
                correct = (top_preds[:, 0] == label).long()
                indices_batch = top_preds.gather(1, correct.view(-1, 1))
                adv_loss_val = (
                    pred[:, label] - pred.gather(1, indices_batch).squeeze() + kappa
                ).clamp(min=0).mean()

            # Similarity constraint
            ref_embeds = coeffs @ self.ref_embeddings[None, :, :]
            ref_output = self.ref_model(inputs_embeds=ref_embeds)

            if lam_sim > 0:
                output = ref_output.hidden_states[embed_layer]
                if constraint.startswith('bertscore'):
                    ref_loss_val = -lam_sim * bert_score(
                        orig_output, output, weights=ref_weights
                    ).mean()
                else:  # Cosine similarity
                    output = output.mean(1)
                    orig = orig_output.mean(1)
                    cosine = (output * orig).sum(1) / output.norm(2, 1) / orig.norm(2, 1)
                    ref_loss_val = -lam_sim * cosine.mean()
            else:
                ref_loss_val = torch.tensor(0.0).to(self.device)

            # Perplexity constraint
            if lam_perp > 0:
                perp_loss_val = lam_perp * log_perplexity(ref_output.logits, coeffs)
            else:
                perp_loss_val = torch.tensor(0.0).to(self.device)

            # Total loss
            total_loss = adv_loss_val + ref_loss_val + perp_loss_val
            total_loss.backward()

            # Zero out gradients for forbidden tokens and forbidden vocabulary
            log_coeffs.grad.index_fill_(0, forbidden_indices, 0)

            # Also zero out gradients for forbidden vocabulary items
            if only_word_substitution:
                log_coeffs.grad[:, ~vocab_mask] = 0

            optimizer.step()

            # Re-apply vocabulary mask after optimization
            if only_word_substitution:
                with torch.no_grad():
                    log_coeffs[:, ~vocab_mask] = -1e10

            # Log losses
            adv_losses.append(adv_loss_val.item())
            ref_losses.append(ref_loss_val.item())
            perp_losses.append(perp_loss_val.item())

            if i % 10 == 0:
                print(f'Iter {i}: adv_loss={adv_loss_val:.4f}, '
                      f'ref_loss={ref_loss_val:.4f}, perp_loss={perp_loss_val:.4f}')

        # Generate adversarial text
        with torch.no_grad():
            best_adv_text = None
            best_adv_logits = None

            for _ in range(gumbel_samples):
                adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)

                # Only keep the content tokens (exclude CLS, SEP, PAD)
                adv_ids_content = adv_ids[content_mask]

                # Decode only the content tokens
                adv_text = self.tokenizer.decode(adv_ids_content, skip_special_tokens=True)

                # Re-tokenize and check prediction
                x = self.tokenizer(
                    adv_text,
                    max_length=256,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)

                adv_logits = self.final_model(**x).logits.cpu()

                if adv_logits.argmax() != label:
                    best_adv_text = adv_text
                    best_adv_logits = adv_logits
                    break

            if best_adv_text is None:
                best_adv_text = adv_text
                best_adv_logits = adv_logits

        return {
            'clean_text': clean_text,
            'adv_text': best_adv_text,
            'clean_logits': clean_logits,
            'adv_logits': best_adv_logits,
            'label': label,
            'success': best_adv_logits.argmax().item() != label,
            'adv_losses': adv_losses,
            'ref_losses': ref_losses,
            'perp_losses': perp_losses
        }


def main():
    parser = argparse.ArgumentParser(description="BERT Adversarial Attack on AG News")
    parser.add_argument("--model_path", default="bert-base-uncased", type=str)
    parser.add_argument("--checkpoint_path", default="bertmod/10.pth", type=str)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--num_iters", default=100, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--lam_sim", default=1, type=float)
    parser.add_argument("--lam_perp", default=1, type=float)
    parser.add_argument("--initial_coeff", default=15, type=float)
    parser.add_argument("--only_word_substitution", default=True, type=bool)
    parser.add_argument("--output_file", default="adversarial_results.pt", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Initialize attacker
    attacker = BertAdversarialAttack(
        model_path=args.model_path,
        use_pretrained=(args.checkpoint_path is None),
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    # Load data
    print("Loading AG News dataset...")
    encoded_dataset, idf_dict = attacker.load_agnews_data(
        num_samples=args.num_samples,
        start_idx=args.start_idx
    )

    # Attack samples
    results = []
    test_data = encoded_dataset['test']

    for idx in range(args.start_idx, min(args.start_idx + args.num_samples, len(test_data))):
        print(f"\n{'='*50}")
        print(f"Attacking sample {idx}")
        print(f"{'='*50}")

        result = attacker.attack_sample(
            input_ids=test_data['input_ids'][idx],
            label=test_data['label'][idx],
            idf_dict=idf_dict,
            num_iters=args.num_iters,
            lr=args.lr,
            lam_sim=args.lam_sim,
            lam_perp=args.lam_perp,
            initial_coeff=args.initial_coeff,
            only_word_substitution=args.only_word_substitution
        )

        results.append(result)

        print(f"\nResults:")
        print(f"Success: {result['success']}")
        print(f"Clean text: {result['clean_text'][:100]}...")
        print(f"Adv text: {result['adv_text'][:100]}...")

    # Calculate statistics
    success_rate = sum(r['success'] for r in results) / len(results)
    print(f"\n{'='*50}")
    print(f"Overall Success Rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()