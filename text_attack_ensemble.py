"""
Ensemble Adversarial attack on BERT models using AG News dataset
Modified to work with ensemble of BERT models
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
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


class EnsembleBertAdversarialAttack:
    def __init__(self,
                 model_paths=None,
                 checkpoint_paths=None,
                 base_model_path='bert-base-uncased',
                 device='cuda'):
        """
        Initialize the ensemble adversarial attack

        Args:
            model_paths: List of paths to BERT models (if different architectures)
            checkpoint_paths: List of checkpoint paths for loading model weights
            base_model_path: Base model path for tokenizer and embeddings
            device: Device to run on
        """
        self.device = device
        self.base_model_path = base_model_path

        # Load tokenizer (shared across all models)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        self.tokenizer.model_max_length = 512

        # Load ensemble of models
        self.models = []

        if checkpoint_paths:
            # Load models from checkpoints
            print(f"Loading ensemble of {len(checkpoint_paths)} models...")
            for i, checkpoint_path in enumerate(checkpoint_paths):
                model_path = model_paths[i] if model_paths and i < len(model_paths) else base_model_path
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=4
                ).to(device)

                if os.path.exists(checkpoint_path):
                    print(f'Loading model {i+1} from checkpoint: {checkpoint_path}')
                    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                else:
                    print(f'Warning: Checkpoint {checkpoint_path} not found, using pretrained weights')

                model.eval()
                self.models.append(model)
        else:
            # Use a single pretrained model
            print("No checkpoints provided, using single pretrained model")
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path, num_labels=4
            ).to(device)
            model.eval()
            self.models = [model]

        print(f"Ensemble loaded with {len(self.models)} models")

        # Load final evaluation model (could be a different model or one from ensemble)
        self.final_model = self.models[-1]  # Use last model as final evaluator

        # Load reference model for perplexity calculation
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            'gpt2', output_hidden_states=True
        ).to(device)
        self.ref_model.eval()

        # Get embeddings (use first model's embeddings as reference)
        with torch.no_grad():
            vocab_size = self.tokenizer.vocab_size
            self.embeddings = self.models[0].get_input_embeddings()(
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
        idf_dict = {}

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

    def ensemble_forward(self, inputs_embeds=None, input_ids=None, attention_mask=None):
        """
        Forward pass through ensemble, returning averaged logits
        """
        ensemble_logits = []

        for model in self.models:
            if inputs_embeds is not None:
                logits = model(inputs_embeds=inputs_embeds).logits
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            ensemble_logits.append(logits)

        # Average logits across ensemble
        return torch.stack(ensemble_logits).mean(dim=0)

    def ensemble_gradient(self, log_coeffs, label, batch_size, adv_loss='ce', kappa=5.0):
        """
        Calculate ensemble gradient by averaging gradients from all models
        Similar to EnsembleMIFGSM
        """
        ensemble_grad = torch.zeros_like(log_coeffs)

        for model in self.models:
            # Create computation graph for this model
            log_coeffs_copy = log_coeffs.clone().requires_grad_(True)

            # Sample from Gumbel-Softmax
            coeffs = F.gumbel_softmax(
                log_coeffs_copy.unsqueeze(0).repeat(batch_size, 1, 1),
                hard=False
            )

            # Get input embeddings using this model's embedding layer
            model_embeddings = model.get_input_embeddings()(
                torch.arange(0, self.tokenizer.vocab_size).long().to(self.device)
            )
            inputs_embeds = coeffs @ model_embeddings[None, :, :]

            # Forward pass through this model
            pred = model(inputs_embeds=inputs_embeds).logits

            # Calculate adversarial loss for this model
            if adv_loss == 'ce':
                loss = -F.cross_entropy(
                    pred,
                    torch.ones(batch_size).long().to(self.device) * label
                )
            else:  # CW loss
                top_preds = pred.sort(descending=True)[1]
                correct = (top_preds[:, 0] == label).long()
                indices_batch = top_preds.gather(1, correct.view(-1, 1))
                loss = (
                    pred[:, label] - pred.gather(1, indices_batch).squeeze() + kappa
                ).clamp(min=0).mean()

            # Get gradient for this model
            grad = torch.autograd.grad(loss, log_coeffs_copy, retain_graph=False)[0]
            ensemble_grad += grad

        # Average gradient across ensemble
        ensemble_grad = ensemble_grad / len(self.models)

        return ensemble_grad

    def attack_sample(self,
                     input_ids,
                     label,
                     idf_dict=None,
                     num_iters=200,
                     lr=0.1,
                     batch_size=10,
                     lam_sim=5.0,
                     lam_perp=2.0,
                     kappa=5.0,
                     initial_coeff=10,
                     constraint='bertscore_idf',
                     adv_loss='ce',
                     gumbel_samples=100,
                     embed_layer=-1,
                     only_word_substitution=True,
                     momentum_decay=0.9):  # Add momentum for ensemble attack
        """
        Attack a single sample using ensemble of models

        Returns:
            Dictionary containing attack results
        """
        input_ids = torch.LongTensor(input_ids).to(self.device)

        # Get clean prediction from ensemble
        with torch.no_grad():
            clean_logits = self.ensemble_forward(
                input_ids=input_ids.unsqueeze(0)
            ).cpu()

        # Decode original text
        clean_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f'Original text: {clean_text}')
        print(f'Label: {label}, Clean ensemble prediction: {clean_logits.argmax().item()}')

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

        # If only_word_substitution, also forbid subword tokens
        if only_word_substitution:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy())
            for i, token in enumerate(tokens):
                if token.startswith('##'):
                    forbidden[i] = True

        forbidden_indices = torch.from_numpy(
            np.arange(0, len(input_ids))[forbidden]
        ).to(self.device)

        # Find content positions
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            first_sep_pos = sep_positions[0].item()
            content_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            content_mask[1:first_sep_pos] = True
        else:
            content_mask = torch.ones_like(input_ids, dtype=torch.bool)
            content_mask[0] = False
            content_mask[input_ids == pad_token_id] = False

        # Create vocabulary mask
        vocab_mask = torch.ones(self.tokenizer.vocab_size, dtype=torch.bool)
        if only_word_substitution:
            for token_id, token in enumerate(self.tokenizer.get_vocab()):
                if token.startswith('##') or token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                    vocab_mask[token_id] = False

        # Initialize coefficients
        log_coeffs = torch.zeros(len(input_ids), self.embeddings.size(0))
        indices = torch.arange(log_coeffs.size(0)).long()
        log_coeffs[indices, input_ids.cpu()] = initial_coeff

        # Apply vocabulary mask
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

        # Initialize momentum (similar to EnsembleMIFGSM)
        momentum = torch.zeros_like(log_coeffs).to(self.device)

        # Attack loop
        adv_losses, ref_losses, perp_losses = [], [], []

        for i in range(num_iters):
            optimizer.zero_grad()

            # Get ensemble gradient
            ensemble_grad_adv = self.ensemble_gradient(
                log_coeffs, label, batch_size, adv_loss, kappa
            )

            # Calculate similarity and perplexity losses
            coeffs = F.gumbel_softmax(
                log_coeffs.unsqueeze(0).repeat(batch_size, 1, 1),
                hard=False
            )

            # Similarity constraint
            ref_embeds = coeffs @ self.ref_embeddings[None, :, :]
            ref_output = self.ref_model(inputs_embeds=ref_embeds)

            if lam_sim > 0:
                output = ref_output.hidden_states[embed_layer]
                if constraint.startswith('bertscore'):
                    ref_loss_val = -lam_sim * bert_score(
                        orig_output, output, weights=ref_weights
                    ).mean()
                else:
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

            # Get gradients for similarity and perplexity
            if lam_sim > 0 or lam_perp > 0:
                total_aux_loss = ref_loss_val + perp_loss_val
                aux_grad = torch.autograd.grad(total_aux_loss, log_coeffs, retain_graph=False)[0]
            else:
                aux_grad = torch.zeros_like(log_coeffs)

            # Combine gradients
            total_grad = ensemble_grad_adv + aux_grad

            # Apply momentum (similar to MIFGSM)
            momentum = momentum_decay * momentum + total_grad / torch.mean(torch.abs(total_grad) + 1e-10)

            # Update with momentum
            with torch.no_grad():
                log_coeffs.data = log_coeffs.data + lr * momentum

                # Zero out forbidden tokens
                log_coeffs.data[forbidden_indices] = log_coeffs.data[forbidden_indices] - lr * momentum[forbidden_indices]

                # Re-apply vocabulary mask
                if only_word_substitution:
                    log_coeffs.data[:, ~vocab_mask] = -1e10

            # Calculate losses for logging
            with torch.no_grad():
                # Get average adversarial loss from ensemble
                coeffs_eval = F.gumbel_softmax(log_coeffs.unsqueeze(0), hard=False)
                inputs_embeds_eval = coeffs_eval @ self.embeddings[None, :, :]
                pred_eval = self.ensemble_forward(inputs_embeds=inputs_embeds_eval)

                if adv_loss == 'ce':
                    adv_loss_val = -F.cross_entropy(
                        pred_eval, torch.ones(1).long().to(self.device) * label
                    ).item()
                else:
                    adv_loss_val = 0.0

            adv_losses.append(adv_loss_val)
            ref_losses.append(ref_loss_val.item() if torch.is_tensor(ref_loss_val) else ref_loss_val)
            perp_losses.append(perp_loss_val.item() if torch.is_tensor(perp_loss_val) else perp_loss_val)

            if i % 10 == 0:
                print(f'Iter {i}: adv_loss={adv_loss_val:.4f}, '
                      f'ref_loss={ref_losses[-1]:.4f}, perp_loss={perp_losses[-1]:.4f}')

        # Generate adversarial text
        with torch.no_grad():
            best_adv_text = None
            best_adv_logits = None

            for _ in range(gumbel_samples):
                adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)

                # Only keep content tokens
                adv_ids_content = adv_ids[content_mask]

                # Decode only content tokens
                adv_text = self.tokenizer.decode(adv_ids_content, skip_special_tokens=True)

                # Re-tokenize and check prediction
                x = self.tokenizer(
                    adv_text,
                    max_length=256,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)

                # Evaluate on final model
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
    parser = argparse.ArgumentParser(description="Ensemble BERT Adversarial Attack on AG News")
    parser.add_argument("--base_model_path", default="bert-base-uncased", type=str)
    parser.add_argument("--checkpoint_dir", default="./text_ens_1", type=str,
                       help="Directory containing ensemble model checkpoints")
    parser.add_argument("--num_models", default=10, type=int,
                       help="Number of models to use from ensemble")
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--num_iters", default=100, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--lam_sim", default=1, type=float)
    parser.add_argument("--lam_perp", default=1, type=float)
    parser.add_argument("--initial_coeff", default=15, type=float)
    parser.add_argument("--momentum_decay", default=0.9, type=float,
                       help="Momentum decay for ensemble attack")
    parser.add_argument("--only_word_substitution", default=True, type=bool)
    parser.add_argument("--output_file", default="ensemble_adversarial_results.pt", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load ensemble model checkpoints
    checkpoint_paths = []
    if os.path.exists(args.checkpoint_dir):
        for i in range(0, args.num_models + 1):
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_{i}.pth')
            if os.path.exists(checkpoint_path):
                checkpoint_paths.append(checkpoint_path)
            else:
                print(f"Warning: {checkpoint_path} not found")

    if not checkpoint_paths:
        print("No checkpoint paths found, using pretrained models")
        checkpoint_paths = None

    # Initialize ensemble attacker
    attacker = EnsembleBertAdversarialAttack(
        checkpoint_paths=checkpoint_paths,
        base_model_path=args.base_model_path,
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
        print(f"Attacking sample {idx} with ensemble of {len(attacker.models)} models")
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
            momentum_decay=args.momentum_decay,
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
    print(f"Ensemble Size: {len(attacker.models)} models")

    # Save results
    torch.save(results, args.output_file)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()