import os
os.environ["HF_HOME"] = "/home/linlab/huggingface_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import glob
import json
import pathlib
import re
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from time import strftime
from time import gmtime
from llama_cpp import Llama
import time
import json_repair
import argparse

SENTENCE_END = re.compile(r'[.?!]$')

"""Utility functions and classes."""
from collections.abc import Generator, Sequence
import copy
import dataclasses
import json
import sys
from typing import Any, List, Optional

import numpy as np
from scipy import optimize

# import numba
import numpy as np


# Edit operations.
_CORRECT = 0
_SUBSTITUTION = 1
_INSERTION = 2
_DELETION = 3


# Computes the Levenshtein alignment between strings ref and hyp, where the
# tokens in each string are separated by delimiter.
# Outputs a tuple : (edit_distance, alignment) where
# alignment is a list of pairs (ref_pos, hyp_pos) where ref_pos is a position
# in ref and hyp_pos is a position in hyp.
# As an example, for strings 'a b' and 'a c', the output would look like:
# (1, [(0,0), (1,1)]
# Note that insertions are represented as (-1, j) and deletions as (i, -1).
# @numba.njit
def levenshtein_with_edits(
    ref: str,
    hyp: str,
    print_debug_info: bool = False) -> tuple[int, list[tuple[int, int]]]:
  align = []
  s1 = ref.split()
  s2 = hyp.split()
  n1 = len(s1)
  n2 = len(s2)
  costs = np.zeros((n1+1, n2+1), dtype=np.int32)
  backptr = np.zeros((n1+1, n2+1), dtype=np.int32)

  for i in range(n1+1):  # ref
    costs[i][0] = i  # deletions

  for j in range(n2):  # hyp
    costs[0][j+1] = j+1  # insertions
    for i in range(n1):  # ref
      # (i,j) <- (i,j-1)
      ins = costs[i+1][j] + 1
      # (i,j) <- (i-1,j)
      del_ = costs[i][j+1] + 1
      # (i,j) <- (i-1,j-1)
      sub = costs[i][j] + (s1[i] != s2[j])
      costs[i + 1][j + 1] = min(ins, del_, sub)
      if (costs[i+1][j+1] == ins):
        backptr[i+1][j+1] = _INSERTION
      elif (costs[i+1][j+1] == del_):
        backptr[i+1][j+1] = _DELETION
      elif (s1[i] == s2[j]):
        backptr[i+1][j+1] = _CORRECT
      else:
        backptr[i+1][j+1] = _SUBSTITUTION

  if print_debug_info:
    print("Mincost: ", costs[n1][n2])
  i = n1
  j = n2
  # Emits pairs (n1_pos, n2_pos) where n1_pos is a position in n1 and n2_pos
  # is a position in n2.
  while (i > 0 or j > 0):
    if print_debug_info:
      print("i: ", i, " j: ", j)
    ed_op = _CORRECT
    if (i >= 0 and j >= 0):
      ed_op = backptr[i][j]
    if (i >= 0 and j < 0):
      ed_op = _DELETION
    if (i < 0 and j >= 0):
      ed_op = _INSERTION
    if (i < 0 and j < 0):
      raise RuntimeError("Invalid alignment")
    if (ed_op == _INSERTION):
      align.append((-1, j-1))
      j -= 1
    elif (ed_op == _DELETION):
      align.append((i-1, -1))
      i -= 1
    else:
      align.append((i-1, j-1))
      i -= 1
      j -= 1

  align.reverse()
  return costs[n1][n2], align

PUNCTUATIONS = [",", ".", "_", "?", "!", "-", '"', "'"]


@dataclasses.dataclass
class PromptOptions:
  """Options for generating prompts."""

  # For prompt segmentation.
  emit_input_length: int = 896
  emit_target_length: int = 896

  # Prefix and suffix for prompt and completion.
  # As a reference, OpenAI finetuning API usually suggests:
  # - No prompt prefix
  # - Prompt suffix: " -> "
  # - Completion suffix: " END"
  prompt_prefix: str = ""
  prompt_suffix: str = " --> "
  completion_suffix: str = ""

  # How do we represent the speaker token.
  # We may consider shorter prefix for token efficiency.
  speaker_prefix: str = "<speaker:"
  speaker_suffix: str = ">"


def normalize_text(text: str) -> str:
  """Normalize text."""
  # Convert to lower case.
  text_lower = text.lower().strip()

  # Remove punctuations.
  words = text_lower.split()
  new_words = []
  for word in words:
    new_word = word
    for punc in PUNCTUATIONS:
      replaced = new_word.replace(punc, "")
      if len(replaced.split()) != 1:
        continue
      new_word = replaced
    new_words.append(new_word)
  return " ".join(new_words)


def speakers_transform(speakers: Sequence[str]) -> list[str]:
  """Transform list of speakers to be order based."""
  spk_map = {}
  index = 0
  for spk in speakers:
    if spk not in spk_map:
      index += 1
      spk_map[spk] = index
  return [str(spk_map[spk]) for spk in speakers]


def get_aligned_hyp_speakers(
    hyp_text: str,
    ref_text: str,
    ref_spk: str,
    print_debug_info: bool = False,
) -> str:
  """Align ref_text to hyp_text, then apply the alignment to ref_spk."""
  # Counters for insertions and deletions in hyp and ref text alignment.
  num_insertions, num_deletions = 0, 0

  # Get the alignment.
  _, align = levenshtein_with_edits(
      normalize_text(ref_text), normalize_text(hyp_text)
  )

  ref_spk_list = ref_spk.split()
  hyp_spk_align = []

  # Apply the alignment on ref speakers.
  for i, j in align:
    if i == -1:
      # hyp has insertion
      hyp_spk_align.append("-1")
      num_insertions += 1
    elif j == -1:
      # hyp has deletion
      num_deletions += 1
      continue
    else:
      hyp_spk_align.append(ref_spk_list[i])
  hyp_spk_align = " ".join(hyp_spk_align)

  if print_debug_info:
    print("Number of insertions: ", num_insertions)
    print("Number of deletions: ", num_deletions)
    # This is not the traditional denominator of WER. Instead, this is
    # len(hyp) + len(ref) - len(SUB).
    print("Length of align pairs: ", len(align))
  return hyp_spk_align


def get_oracle_speakers(hyp_spk: str, hyp_spk_align: str) -> Sequence[int]:
  """Get the oracle speakers for hypothesis."""
  hyp_spk_list = [int(x) for x in hyp_spk.split()]
  hyp_spk_align_list = [int(x) for x in hyp_spk_align.split()]

  # Build cost matrix.
  max_spk = max(max(hyp_spk_list), max(hyp_spk_align_list))
  cost_matrix = np.zeros((max_spk, max_spk))
  for aligned, original in zip(hyp_spk_align_list, hyp_spk_list):
    cost_matrix[aligned - 1, original - 1] += 1

  # Solve alignment.
  row_index, col_index = optimize.linear_sum_assignment(
      cost_matrix, maximize=True
  )

  # Build oracle.
  hyp_spk_oracle = hyp_spk_list.copy()
  for i in range(len(hyp_spk_list)):
    if hyp_spk_align_list[i] == -1:
      # There are some missing words. In such cases, we just use the original
      # speaker for these words if possible.
      if hyp_spk_list[i] == -1:
        # If we don't have original speaker for missing words, just use the
        # previous speaker if possible.
        # This is useful for the update_hyp_text_in_utt_dict() function.
        if i == 0:
          hyp_spk_oracle[i] = 1
        else:
          hyp_spk_oracle[i] = hyp_spk_oracle[i - 1]
      continue
    assert row_index[hyp_spk_align_list[i] - 1] == hyp_spk_align_list[i] - 1
    hyp_spk_oracle[i] = col_index[hyp_spk_align_list[i] - 1] + 1

  return hyp_spk_oracle


# Transcript-Preserving Speaker Transfer (TPST)
def transcript_preserving_speaker_transfer(
    src_text: str, src_spk: str, tgt_text: str, tgt_spk: str
) -> str:
  """Apply source speakers to target."""
  if len(tgt_text.split()) != len(tgt_spk.split()):
    raise ValueError("tgt_text and tgt_spk must have the same length")
  if len(src_text.split()) != len(src_spk.split()):
    raise ValueError("src_text and src_spk must have the same length")
  tgt_spk_align = get_aligned_hyp_speakers(
      hyp_text=tgt_text,
      ref_text=src_text,
      ref_spk=src_spk,
  )
  oracle_speakers = get_oracle_speakers(
      hyp_spk=tgt_spk, hyp_spk_align=tgt_spk_align
  )
  return " ".join([str(x) for x in oracle_speakers])


# We can use this to finetune LLM.
# Inputs (prompts): hyp diarized text
# Targets: hyp diarized text with oracle speakers
def ref_to_oracle(json_dict: dict[str, str]) -> str:
  """Apply reference speakers to hypothesis."""
  return transcript_preserving_speaker_transfer(
      src_text=json_dict["ref_text"],
      src_spk=json_dict["ref_spk"],
      tgt_text=json_dict["hyp_text"],
      tgt_spk=json_dict["hyp_spk"],
  )


# Similar to ref_to_oracle, but the opposite direction.
# We can use this to finetune LLM.
# Inputs (prompts): ref diarized text with degraded speakers
# Targets: ref diarized text
def hyp_to_degraded(json_dict: dict[str, str]) -> str:
  """Apply hypothesis speakers to reference."""
  return transcript_preserving_speaker_transfer(
      src_text=json_dict["hyp_text"],
      src_spk=json_dict["hyp_spk"],
      tgt_text=json_dict["ref_text"],
      tgt_spk=json_dict["ref_spk"],
  )


def create_diarized_text(
    word_labels: Sequence[str],
    speaker_labels: Sequence[str],
    use_new_line: bool = False,
    po: PromptOptions = PromptOptions(),
) -> str:
  """Create diarized text from words and speaker labels."""
  output = []
  previous_speaker = None
  for word, speaker in zip(word_labels, speaker_labels):
    if speaker != previous_speaker:
      if previous_speaker and use_new_line:
        output.append("\n")
      output.append(po.speaker_prefix + speaker + po.speaker_suffix)
    output.append(word)
    previous_speaker = speaker
  return " ".join(output)


def extract_text_and_spk(
    completions: str, po: PromptOptions, skip_meaningless_speaker: bool = True
) -> tuple[str, str]:
  """Extract the text and spk from the completions string."""
  spk = "1"
  previous_spk = "1"
  result_text = []
  result_spk = []
  for word in completions.split():
    if word.startswith(po.speaker_prefix):
      if not word.endswith(po.speaker_suffix):
        word += po.speaker_suffix
      spk = word[len(po.speaker_prefix):-len(po.speaker_suffix)]
      # Handle undefined behaviors of non-recognizable spk with a placeholder.
      try:
        spk_int = int(spk)
        if not spk or spk_int < 1 or spk_int > 10:
          raise ValueError("Seeing unexpected word: ", word)
        previous_spk = spk
      except ValueError as exc:
        if skip_meaningless_speaker:
          print("Skipping meaningless speaker token:", word)
          spk = previous_spk
        else:
          raise exc
    else:
      result_text.append(word)
      result_spk.append(spk)
  return " ".join(result_text), " ".join(result_spk)


def discard_empty_str_and_remove_boundary_white_space(
    inputs: List[str],
) -> List[str]:
  return [x.strip() for x in inputs if x.strip()]


@dataclasses.dataclass
class JsonUtteranceReader:
  """Read the json files and generate prompts and targets."""

  json_files: str  # Ignored if utt is given.
  text_field: str
  input_speaker_field: str
  target_speaker_field: str  # If not given, will skip targets.
  po: PromptOptions
  utt: dict[str, str] = dataclasses.field(default_factory=dict)

  def generate_utts(self) -> Generator[dict[str, str], None, None]:
    """Generate an utterance from all json files."""
    if self.utt:
      yield self.utt
      return

    for json_file in self.json_files.split(","):
      with open(json_file) as f:
        data_dict = json.load(f)
        for utt in data_dict["utterances"]:
          yield utt

  def generate_data_tuple(self) -> Generator[tuple[str, str, str], None, None]:
    """Generate uttid-prompt-target tuples."""
    for utt in self.generate_utts():
      yield from self.generate_data_tuple_for_utt(utt)

  def generate_data_dict(self) -> Generator[dict[str, str], None, None]:
    """Generate a dict that can be used for datasets.Dataset.from_generator."""
    for uttid, prompt, target in self.generate_data_tuple():
      yield {"uttid": uttid, "prompt": prompt, "target": target}

  def generate_data_tuple_for_utt(
      self, utt: dict[str, str]
  ) -> Generator[tuple[str, str, str], None, None]:
    """Generate uttid-prompt-target tuples from a single utterance."""
    self.seg_id = 0
    utt_id = utt["utterance_id"]

    # Get the fields from the utterance.
    words = discard_empty_str_and_remove_boundary_white_space(
        utt[self.text_field].split(" ")
    )
    p_speakers = discard_empty_str_and_remove_boundary_white_space(
        utt[self.input_speaker_field].split(" ")
    )
    assert len(words) == len(p_speakers)
    if self.target_speaker_field:
      t_speakers = discard_empty_str_and_remove_boundary_white_space(
          utt[self.target_speaker_field].split(" ")
      )
      assert len(words) == len(t_speakers)
    else:
      t_speakers = []

    yield from self.generate_data_tuple_from_range(
        utt_id, words, p_speakers, t_speakers, start=0, end=len(words)
    )

  def generate_data_tuple_from_range(
      self, utt_id, words, p_speakers, t_speakers, start, end
  ) -> Generator[tuple[str, str, str], None, None]:
    """Generate uttid-prompt-target tuples from a range of words."""
    # Decide whether to call recursively from the estimated length.
    estimated_prompt_length = (
        len(self.po.prompt_prefix)
        + len(" ".join(words[start:end]))
        + len(self.po.prompt_suffix)
    )
    if (
        estimated_prompt_length > self.po.emit_input_length
        or estimated_prompt_length > self.po.emit_target_length
    ):
      yield from self.generate_data_tuple_from_range(
          utt_id, words, p_speakers, t_speakers, start, (start + end) // 2
      )
      yield from self.generate_data_tuple_from_range(
          utt_id, words, p_speakers, t_speakers, (start + end) // 2, end
      )
      return

    prompt = self.po.prompt_prefix
    previous_p_spk = ""
    target = ""
    previous_t_spk = ""

    # Main loop.
    for i in range(start, end):
      word = words[i]
      p_spk = p_speakers[i]
      if p_spk != previous_p_spk:
        if previous_p_spk:
          prompt += " "
        prompt += self.po.speaker_prefix + p_spk + self.po.speaker_suffix
      prompt += " " + word
      previous_p_spk = p_spk

      if self.target_speaker_field:
        t_spk = t_speakers[i]
        if t_spk != previous_t_spk:
          if previous_t_spk:
            target += " "
          target += self.po.speaker_prefix + t_spk + self.po.speaker_suffix
        target += " " + word
        previous_t_spk = t_spk

    prompt_id = utt_id + "_seg" + str(self.seg_id)
    prompt += self.po.prompt_suffix
    target += self.po.completion_suffix
    if (
        len(prompt) <= self.po.emit_input_length
        and len(target) <= self.po.emit_target_length
    ):
      yield (prompt_id, prompt, target)
      self.seg_id += 1
    else:
      yield from self.generate_data_tuple_from_range(
          utt_id, words, p_speakers, t_speakers, start, (start + end) // 2
      )
      yield from self.generate_data_tuple_from_range(
          utt_id, words, p_speakers, t_speakers, (start + end) // 2, end
      )


def generate_prompts(
    utt: dict[str, str],
    po: PromptOptions,
    text_field: str = "hyp_text",
    input_speaker_field: str = "hyp_spk",
) -> list[str]:
  """Generate a list of prompts for a given utt."""
  po_modified = copy.deepcopy(po)
  po_modified.emit_target_length = sys.maxsize
  reader = JsonUtteranceReader(
      json_files="",
      text_field=text_field,
      input_speaker_field=input_speaker_field,
      target_speaker_field="",
      po=po_modified,
      utt=utt,
  )
  prompts = []
  for _, prompt, _ in reader.generate_data_tuple():
    prompts.append(prompt)
  if len(prompts) > 1:
    for prompt in prompts:
      if len(prompt) < po.emit_input_length / 3:
        raise RuntimeError("Prompt too short: ", prompt)
  return prompts


def find_utt_dict(
    utt_id: str, data_dict: dict[str, Any]
) -> Optional[dict[str, str]]:
  """Find a utt_dict with a speicifc utterance_id from data_dict."""
  for utt_dict in data_dict["utterances"]:
    if utt_dict["utterance_id"] == utt_id:
      return utt_dict
  return None


def update_hyp_text_in_utt_dict(
    input_utt_dict: dict[str, str], new_hyp_text
) -> dict[str, str]:
  """Update the hyp_text of a json utt_dict.

  We also transfer its original hyp_spk to the new hyp_text.

  This is useful if we want to use USM ASR transcripts to replace the
  turn-to-diarize transcripts, as the WER of turn-to-diarize transcripts is too
  high.

  Args:
    input_utt_dict: the input utt_dict
    new_hyp_text: the new hyp_text

  Returns:
    the new utt_dict
  """
  utt_dict = copy.deepcopy(input_utt_dict)
  # We don't know the speakers for new_hyp_text, so just use -1 as initial
  # speakers.
  new_hyp_spk = transcript_preserving_speaker_transfer(
      src_text=utt_dict["hyp_text"],
      src_spk=utt_dict["hyp_spk"],
      tgt_text=new_hyp_text,
      tgt_spk=" ".join(["-1" for _ in new_hyp_text.split()]),
  )
  # Update the utt_dict.
  utt_dict["hyp_text"] = new_hyp_text
  utt_dict["hyp_spk"] = new_hyp_spk
  utt_dict["hyp_diarized_text"] = create_diarized_text(
      new_hyp_text.split(), new_hyp_spk.split()
  )
  return utt_dict


def truncate_suffix_and_tailing_text(text: str, suffix: str) -> str:
  """Tailing text after suffix should be removed as well."""
  if suffix and suffix in text:
    return text[: text.find(suffix)]
  return text


def postprocess_completions_for_utt(
    utt: dict[str, Any],
    llm_text_field: str = "llm_text",
    llm_speaker_field: str = "llm_spk",
    transfered_llm_speaker_field: str = "hyp_spk_llm",
    hyp_text_field: str = "hyp_text",
    hyp_spk_field: str = "hyp_spk",
    po: PromptOptions = PromptOptions(),
) -> None:
  """Postprocess the LLM completions of an utterance json dict."""
  # Remove completion suffix if it exists.
  completions_list = []
  for completion in utt["completions"]:
    if po.completion_suffix and po.completion_suffix in completion:
      completion = truncate_suffix_and_tailing_text(
          completion, po.completion_suffix
      )
    completions_list.append(completion)
  completions = " ".join(completions_list).strip()

  # Extract text and speaker.
  utt[llm_text_field], utt[llm_speaker_field] = extract_text_and_spk(
      completions, po=po
  )
  # Tha TPST alignment on LLM output against recognized hypothesis text can be
  # considered as a postprocessing step to ensure the hypothesis text does not
  # change too much from the diarization baseline.
  # Note: this step can arguably be skipped and we directly use LLM output
  # for evaluation. The assumption is LLM does not change original text too
  # much. `update_sstable_with_speakers` should be updated accordingly if so.
  utt[transfered_llm_speaker_field] = transcript_preserving_speaker_transfer(
      src_text=utt[llm_text_field],
      src_spk=utt[llm_speaker_field],
      tgt_text=utt[hyp_text_field],
      tgt_spk=utt[hyp_spk_field],
  )


def transfer_llm_completion(
    llm_completion: str,
    hyp: str,
    po: PromptOptions = PromptOptions(),
) -> str:
  """Transfer the LLM completion text to use text from hypothesis."""
  llm_text, llm_speaker = extract_text_and_spk(
      llm_completion, po=po
  )
  hyp_text, hyp_speaker = extract_text_and_spk(
      hyp, po=po
  )
  transfered_llm_speaker = transcript_preserving_speaker_transfer(
      src_text=llm_text,
      src_spk=llm_speaker,
      tgt_text=hyp_text,
      tgt_spk=hyp_speaker,
  )
  transferred = create_diarized_text(
      word_labels=hyp_text.split(),
      speaker_labels=transfered_llm_speaker.split(),
      po=po,
  )
  return transferred

def group_into_sentences(words):
    sentences = []
    current = []
    for word in words:
        current.append(word)
        if SENTENCE_END.search(word['text']):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences

def load_rttm(rttm_path):
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append({
                'start': start,
                'end': start + dur,
                'speaker': speaker
            })
    return segments

def deprecated_load_transcript(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Flatten all words from all segments
    all_words = []
    for segment in data.get('segments', []):
        if segment['compression_ratio'] < 4.0 and (segment['end']-segment['start']>0.03): #Filter out obvious hallucinations
            if 'words' in segment:
                all_words.extend(segment['words'])
    
    return all_words

def load_transcript(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_words = []
    
    # Check if this is the new format (with segments array at top level containing segment_number)
    if 'segments' in data and len(data['segments']) > 0:
        # Check if first segment has 'segment_number' key (new format)
        if 'segment_number' in data['segments'][0]:
            # New format: iterate through top-level segments
            for top_segment in data['segments']:
                transcription = top_segment.get('transcription', {})
                # Now process the Whisper segments within each transcription
                for segment in transcription.get('segments', []):
                    if segment['compression_ratio'] < 4.0 and (segment['end'] - segment['start'] > 0.03):
                        if 'words' in segment:
                            all_words.extend(segment['words'])
        else:
            # Old format: segments directly contain Whisper data
            for segment in data.get('segments', []):
                if segment['compression_ratio'] < 4.0 and (segment['end'] - segment['start'] > 0.03):
                    if 'words' in segment:
                        all_words.extend(segment['words'])
    
    return all_words

def print_speaker_turns(speaker_turns, file_path=None):
    lines = []
    for turn in speaker_turns:
        header = f"{turn['speaker']} [{turn['start']:.2f} - {turn['end']:.2f}]"
        lines.append(header)
        for s in turn['sentences']:
            sentence_line = f"{s['text']}"
            lines.append(sentence_line)
        lines.append("")  # blank line between turns

    output = "\n".join(lines)

    if file_path:
        with open(file_path, 'w') as f:
            f.write(output)
    else:
        print(output)

def align_words_to_speakers(words, segments, buffer=0.05):
    speaker_turns = []
    current_turn = None
    
    # Handle edge case: no segments
    if not segments:
        # Assign all words to a default speaker
        if words:
            current_turn = {
                'speaker': 'SPEAKER_00',  # Default speaker label
                'start': words[0]['start'],
                'end': words[-1]['end'],
                'words': words
            }
            speaker_turns.append(current_turn)
        
        # Process sentences and return early
        for turn in speaker_turns:
            turn['sentences'] = []
            for sentence_words in group_into_sentences(turn['words']):
                sent_text = " ".join(w['text'] for w in sentence_words)
                sent_start = sentence_words[0]['start']
                sent_end = sentence_words[-1]['end']
                turn['sentences'].append({
                    'text': sent_text,
                    'start': sent_start,
                    'end': sent_end,
                    'words': sentence_words
                })
        return speaker_turns
    
    # Handle edge case: no words
    if not words:
        return speaker_turns

    def overlaps(word, segment):
        """Check if a word overlaps a segment, with small tolerance buffer."""
        return not (word['end'] < segment['start'] - buffer or 
                    word['start'] > segment['end'] + buffer)

    for w in words:
        # Find overlapping segments (with buffer)
        candidates = [s for s in segments if overlaps(w, s)]

        if candidates:
            # If multiple overlaps, choose the one with greatest overlap duration
            def overlap_amount(s):
                return max(0.0, min(w['end'], s['end']) - max(w['start'], s['start']))
            chosen = max(candidates, key=overlap_amount)
        else:
            # Fallback: assign to nearest segment in time
            def distance(s):
                if w['end'] < s['start']:
                    return s['start'] - w['end']
                elif w['start'] > s['end']:
                    return w['start'] - s['end']
                return 0.0
            chosen = min(segments, key=distance)

        # Continue existing turn if same speaker
        if current_turn and current_turn['speaker'] == chosen['speaker']:
            current_turn['end'] = max(current_turn['end'], w['end'])
            current_turn['words'].append(w)
        else:
            # Save previous turn
            if current_turn:
                speaker_turns.append(current_turn)
            # Start new turn
            current_turn = {
                'speaker': chosen['speaker'],
                'start': w['start'],
                'end': w['end'],
                'words': [w]
            }

    # Add final turn
    if current_turn:
        speaker_turns.append(current_turn)

    for turn in speaker_turns:
        turn['sentences'] = []
        for sentence_words in group_into_sentences(turn['words']):
            sent_text = " ".join(w['text'] for w in sentence_words)
            sent_start = sentence_words[0]['start']
            sent_end = sentence_words[-1]['end']
            turn['sentences'].append({
                'text': sent_text,
                'start': sent_start,
                'end': sent_end,
                'words': sentence_words
            })

    return speaker_turns


def flatten_speaker_turns(speaker_turns):
    output = []
    last_speaker = None

    for turn in speaker_turns:
        speaker = turn['speaker']
        # Use something like <spk 1>, <spk 2>, etc.
        speaker_label = f"<speaker:{int(speaker.split('_')[-1]) + 1}>"

        # Only insert speaker label if speaker changed
        if speaker != last_speaker:
            output.append(speaker_label)
            last_speaker = speaker

        # Append words
        words_text = " ".join(w['text'] for w in turn['words'])
        output.append(words_text)

    return " ".join(output)

def parse_diarizationlm_output(diarizationlm_text: str) -> List[Dict[str, Any]]:
    """
    Parse DiarizationLM output to extract speaker segments and their text.
    
    Args:
        diarizationlm_text: Raw output from DiarizationLM
        
    Returns:
        List of dictionaries with 'speaker', 'text', and 'words' keys
    """
    segments = []
    
    # Find all speaker segments using regex
    pattern = r'<speaker:(\d+)>\s*([^<]*?)(?=<speaker:|\Z)'
    matches = re.findall(pattern, diarizationlm_text, re.DOTALL)
    
    for speaker_id, text in matches:
        text = text.strip()
        if text:  # Only add non-empty segments
            words = text.split()
            segments.append({
                'speaker': f'speaker:{speaker_id}',
                'text': text,
                'words': words
            })
    
    return segments

def create_chronological_word_list(speaker_turns: List[Dict]) -> List[Dict]:
    """
    Create a chronologically sorted list of all words with their metadata.
    
    Args:
        speaker_turns: List of speaker turn dictionaries with words
        
    Returns:
        Chronologically sorted list of word dictionaries
    """
    all_words = []
    
    for turn in speaker_turns:
        speaker = turn['speaker']
        for word_info in turn['words']:
            word_dict = {
                'text': word_info['text'],
                'start': word_info['start'],
                'end': word_info['end'],
                'confidence': word_info['confidence'],
                'original_speaker': speaker
            }
            all_words.append(word_dict)
    
    # Sort by start time to ensure chronological order
    all_words.sort(key=lambda x: x['start'])
    
    return all_words

def normalize_word(word: str) -> str:
    """Normalize word for matching by removing punctuation and converting to lowercase."""
    return word.lower().strip('.,!?;:"\'()[]{}')

def word_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity between two words.
    Returns 1.0 for exact match, partial scores for similar words, 0.0 for no match.
    """
    w1_norm = normalize_word(word1)
    w2_norm = normalize_word(word2)
    
    if w1_norm == w2_norm:
        return 1.0
    
    # Handle empty strings
    if not w1_norm or not w2_norm:
        return 0.0
    
    # Check if one word contains the other (for contractions, etc.)
    if len(w1_norm) > 2 and len(w2_norm) > 2:
        if w1_norm in w2_norm or w2_norm in w1_norm:
            return 0.8
    
    # Simple character-based similarity for very similar words
    if len(w1_norm) > 3 and len(w2_norm) > 3:
        common_chars = set(w1_norm) & set(w2_norm)
        total_chars = set(w1_norm) | set(w2_norm)
        char_similarity = len(common_chars) / len(total_chars) if total_chars else 0
        
        if char_similarity > 0.7:
            return char_similarity * 0.6
    
    return 0.0

def dtw_alignment(diarization_words: List[str], chronological_words: List[Dict]) -> List[Tuple[int, int]]:
    """
    Perform Dynamic Time Warping to align diarization words with chronological words.
    
    Args:
        diarization_words: List of words from DiarizationLM output
        chronological_words: List of word dictionaries with timestamps
        
    Returns:
        List of tuples (diarization_index, chronological_index) representing the alignment
    """
    n, m = len(diarization_words), len(chronological_words)
    
    if n == 0 or m == 0:
        return []
    
    # Fast equality check - if lengths match and all words are identical, return 1:1 alignment
    if n == m:
        # Extract text from chronological_words for comparison
        print('Exact String Length Match')
        chronological_texts = [word_info['text'] for word_info in chronological_words]
        
        # Check if all words match exactly (case-sensitive)
        if diarization_words == chronological_texts:
            # Return 1:1 alignment - each index maps to itself
            return [(i, i) for i in range(n)]
        
        # Optional: Check case-insensitive equality if exact match fails
        if [word.lower() for word in diarization_words] == [text.lower() for text in chronological_texts]:
            return [(i, i) for i in range(n)]

    # Create similarity matrix
    similarity_matrix = np.zeros((n, m))
    for i, d_word in enumerate(diarization_words):
        for j, c_word_info in enumerate(chronological_words):
            similarity_matrix[i, j] = word_similarity(d_word, c_word_info['text'])
    
    # DTW cost matrix (we use 1 - similarity as cost)
    cost_matrix = 1.0 - similarity_matrix
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i-1, j-1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # deletion
                dtw_matrix[i, j-1],     # insertion
                dtw_matrix[i-1, j-1]    # match
            )
    
    # Backtrack to find optimal path
    path = []
    i, j = n, m
    
    while i > 0 and j > 0:
        # Find the direction that led to current cell
        current_cost = dtw_matrix[i, j]
        diagonal_cost = dtw_matrix[i-1, j-1] + cost_matrix[i-1, j-1]
        up_cost = dtw_matrix[i-1, j] + cost_matrix[i-1, j-1]
        left_cost = dtw_matrix[i, j-1] + cost_matrix[i-1, j-1]
        
        if abs(diagonal_cost - current_cost) < 1e-10:
            path.append((i-1, j-1))
            i, j = i-1, j-1
        elif abs(up_cost - current_cost) < 1e-10:
            # Deletion - move up (skip diarization word)
            i = i-1
        else:
            # Insertion - move left (skip chronological word)
            j = j-1
    
    path.reverse()
    return path

def align_with_dtw(diarizationlm_output: str, speaker_turns: List[Dict]) -> List[Dict]:
    """
    Align DiarizationLM output with word-level timestamps using Dynamic Time Warping.
    
    Args:
        diarizationlm_output: Raw output from DiarizationLM
        speaker_turns: List of speaker turn dictionaries with word-level timestamps
        
    Returns:
        List of aligned speaker turns with corrected timestamps
    """
    # Parse the DiarizationLM output
    diarization_segments = parse_diarizationlm_output(diarizationlm_output)
    
    # Create chronologically sorted word list
    chronological_words = create_chronological_word_list(speaker_turns)
    
    if not chronological_words:
        return []
    
    # Option 1: Global DTW alignment for all words at once
    print("Performing global DTW alignment...")
    all_diarization_words = []
    segment_boundaries = []  # Track which words belong to which segment
    
    for segment in diarization_segments:
        start_idx = len(all_diarization_words)
        all_diarization_words.extend(segment['words'])
        end_idx = len(all_diarization_words)
        segment_boundaries.append((start_idx, end_idx, segment))
    
    # Perform global DTW alignment
    global_alignment = dtw_alignment(all_diarization_words, chronological_words)
    
    if not global_alignment:
        print("Global DTW failed, falling back to segment-by-segment approach...")
        return align_with_dtw_segments(diarization_segments, chronological_words)
    
    # Map global alignment back to segments
    aligned_turns = []
    
    for start_idx, end_idx, segment in segment_boundaries:
        # Find alignments for this segment
        segment_alignments = [
            (seg_idx, chron_idx) for seg_idx, chron_idx in global_alignment
            if start_idx <= seg_idx < end_idx
        ]
        
        if not segment_alignments:
            print(f"Warning: No alignments found for segment: {segment['text'][:50]}...")
            continue
        
        # Extract aligned words with timestamps
        aligned_word_infos = []
        
        for seg_idx, chron_idx in segment_alignments:
            if chron_idx < len(chronological_words):
                word_info = chronological_words[chron_idx]
                diarization_word = all_diarization_words[seg_idx]
                aligned_word_infos.append({
                    'text': diarization_word,
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'confidence': word_info['confidence'],
                    'similarity': word_similarity(diarization_word, word_info['text'])
                })
        
        if aligned_word_infos:
            # Calculate segment boundaries
            start_time = aligned_word_infos[0]['start']
            end_time = aligned_word_infos[-1]['end']
            
            # Map speaker format (speaker:3 -> SPEAKER_03)
            # speaker_num = segment['speaker'].split(':')[1]
            # formatted_speaker = f"SPEAKER_{speaker_num.zfill(2)}"
            formatted_speaker = f"<{segment['speaker']}>"
            
            # Calculate average confidence and similarity
            avg_confidence = np.mean([w['confidence'] for w in aligned_word_infos])
            avg_similarity = np.mean([w['similarity'] for w in aligned_word_infos])
            
            aligned_turn = {
                'speaker': formatted_speaker,
                'start': start_time,
                'end': end_time,
                'text': segment['text'],
                'words': aligned_word_infos,
                'confidence': avg_confidence,
                'alignment_quality': avg_similarity
            }
            
            aligned_turns.append(aligned_turn)
    
    return aligned_turns

def align_with_dtw_segments(diarization_segments: List[Dict], chronological_words: List[Dict]) -> List[Dict]:
    """
    Fallback method: align each segment individually with larger search windows.
    """
    aligned_turns = []
    last_used_index = 0
    
    for segment in diarization_segments:
        segment_words = segment['words']
        if not segment_words:
            continue
        
        # Use a much larger search window that doesn't advance aggressively
        window_start = max(0, last_used_index - 10)  # Allow some backtrack
        window_size = len(segment_words) * 5 + 50  # Much larger window
        search_window = chronological_words[window_start:window_start + window_size]
        
        if not search_window:
            print(f"Warning: No search window available for segment: {segment['text'][:50]}...")
            continue
        
        # Perform DTW alignment for this segment
        alignment_path = dtw_alignment(segment_words, search_window)
        
        if not alignment_path:
            print(f"Warning: No alignment found for segment: {segment['text'][:50]}...")
            continue
        
        # Extract aligned words with timestamps
        aligned_word_infos = []
        max_used_index = last_used_index
        
        for seg_idx, chron_idx in alignment_path:
            if chron_idx < len(search_window):
                actual_chron_idx = window_start + chron_idx
                word_info = chronological_words[actual_chron_idx]
                aligned_word_infos.append({
                    'text': segment_words[seg_idx],
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'confidence': word_info['confidence'],
                    'similarity': word_similarity(segment_words[seg_idx], word_info['text'])
                })
                max_used_index = max(max_used_index, actual_chron_idx)
        
        if aligned_word_infos:
            # Calculate segment boundaries
            start_time = aligned_word_infos[0]['start']
            end_time = aligned_word_infos[-1]['end']
            
            # Update last used index more conservatively
            last_used_index = max(last_used_index, max_used_index - len(segment_words) // 2)
            
            # Map speaker format
            # speaker_num = segment['speaker'].split(':')[1]
            # formatted_speaker = f"SPEAKER_{speaker_num.zfill(2)}"
            formatted_speaker = f"<{segment['speaker']}>"
            
            # Calculate averages
            avg_confidence = np.mean([w['confidence'] for w in aligned_word_infos])
            avg_similarity = np.mean([w['similarity'] for w in aligned_word_infos])
            
            aligned_turn = {
                'speaker': formatted_speaker,
                'start': start_time,
                'end': end_time,
                'text': segment['text'],
                'words': aligned_word_infos,
                'confidence': avg_confidence,
                'alignment_quality': avg_similarity
            }
            
            aligned_turns.append(aligned_turn)
    
    return aligned_turns

def print_alignment_debug(diarizationlm_output: str, speaker_turns: List[Dict]):
    """Debug function to analyze the alignment process."""
    diarization_segments = parse_diarizationlm_output(diarizationlm_output)
    chronological_words = create_chronological_word_list(speaker_turns)
    
    print(f"=== ALIGNMENT DEBUG ===")
    print(f"Total chronological words: {len(chronological_words)}")
    print(f"Time range: {chronological_words[0]['start']:.2f}s - {chronological_words[-1]['end']:.2f}s")
    print(f"Total diarization segments: {len(diarization_segments)}")
    
    all_diarization_words = []
    for i, segment in enumerate(diarization_segments):
        word_count = len(segment['words'])
        all_diarization_words.extend(segment['words'])
        print(f"Segment {i+1} ({segment['speaker']}): {word_count} words - '{segment['text'][:50]}...'")
    
    print(f"Total diarization words: {len(all_diarization_words)}")
    print(f"Word ratio: {len(all_diarization_words)}/{len(chronological_words)} = {len(all_diarization_words)/len(chronological_words):.2f}")

def print_aligned_turns_detailed(aligned_turns: List[Dict]):
    """Print aligned turns with detailed information for debugging."""
    total_time_covered = 0
    for i, turn in enumerate(aligned_turns):
        quality = turn.get('alignment_quality', 0)
        duration = turn['end'] - turn['start']
        total_time_covered += duration
        
        # print(f"{turn['speaker']} [{turn['start']:.3f} - {turn['end']:.3f}] ")
        print(f"{turn['speaker']} [{strftime("%H:%M:%S",gmtime(turn['start']))} - {strftime("%H:%M:%S",gmtime(turn['end']))}] ")
            #   f"({duration:.3f}s, Quality: {quality:.2f})")
        print(f"{turn['text']}\n")
    
    print(f"Total time covered by aligned turns: {total_time_covered:.3f}s")

def make_aligned_turns_list_postdiarlm(aligned_turns: List[Dict]):
    """Print aligned turns with detailed information for debugging."""
    total_time_covered = 0
    output = []
    for i, turn in enumerate(aligned_turns):
        duration = turn['end'] - turn['start']
        total_time_covered += duration
        
        # print(f"{turn['speaker']} [{turn['start']:.3f} - {turn['end']:.3f}] ")
        output.append(f"{turn['speaker']} [{strftime("%H:%M:%S",gmtime(turn['start']))} - {strftime("%H:%M:%S",gmtime(turn['end']))}]\n{turn['text']}")
        # output.append(f"{turn['text']}\n")
        
    return output

def split_on_spk_tokens(text):
    # Split on speaker tags and reconstruct with speaker + content
    parts = re.split(r'(<speaker:\d+>)', text)
    result = []

    for i in range(1, len(parts), 2):  # Start at 1, step by 2 to get speaker tags
        if i+1 < len(parts):
            speaker = parts[i]
            content = parts[i+1].strip()
            if content:  # Only add if there's actual content
                result.append(f"{speaker} {content}")
    return result



def process_files(parent_paths):
  attribution_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

  You are labeling roles in a clinical dialogue. 
  Identify which <speaker:N> is the patient and which is the clinician. 
  Be precise and concise.

  Heuristics (use minimally, prefer direct evidence):
  • Patient: reports own symptoms/history, describes feelings/home care/med adherence.
  • Clinician: asks targeted questions, gives instructions/diagnoses, orders tests, interprets results, uses clinical terminology.

  Rules:
  • If uncertain, set the field to null (don't guess).
  • Quote only the shortest snippets needed as evidence.
  • Return valid JSON only (no extra text/markdown).

  ANALYSIS PROCESS:
  1. First, analyze the transcript and identify evidence for each speaker
  2. Document your rationale for the assignment
  3. If you have concerns or confidence is low, explicitly note them
  4. Only then make your final assignment

  Output JSON exactly in this shape:
  {{
  "rationale": "≤50 words explaining your reasoning based on the evidence.",
  "concerns": "Any specific uncertainties, ambiguities, or reasons for low confidence (≤50 words). Set to null if confident.",
  "patient": "speaker:N" | null,
  "clinician": "speaker:N" | null,
  "confidence": 0.0-1.0,
  "evidence": [
  {{"speaker": "speaker:N", "quote": "short quote", "why": "what this shows"}},
  {{"speaker": "speaker:N", "quote": "short quote", "why": "what this shows"}}
  ]
  }}

  Always provide rationale first. If uncertain (confidence < 0.7), document concerns before making assignments.<|eot_id|><|start_header_id|>user<|end_header_id|>

  Transcript:
  <
  {transcript_content}
  >>><|eot_id|><|start_header_id|>assistant<|end_header_id|>

  """

  MODEL_PATH = "/home/linlab/huggingface_cache/diarizationlm-quantized/DiarizationLM-8b-Fisher-v2.f16.gguf"
  llm = Llama(
      model_path=MODEL_PATH,
      n_gpu_layers=-1,  # offload all layers to GPU
      n_ctx=4096,
      n_threads=16,      # adjust based on your CPU
      n_batch=512       # increase for faster inference if enough GPU memory
  )

  LLAMA_PATH = "/home/linlab/huggingface_cache/llama3.3-quantized/Llama-3.3-70B-Instruct.Q4_K_M.gguf"
  new_llm = Llama(
        model_path=LLAMA_PATH,
        echo=False,
        n_gpu_layers=-1,  # offload all layers to GPU
        n_ctx=16384,
        n_threads=16,      # adjust based on your CPU
        n_batch=512       # increase for faster inference if enough GPU memory
    )

  # Usage example:
  def create_attribution_prompt(transcript):
      return attribution_prompt.format(transcript_content=transcript)

  transcript_dir = parent_paths
  transcript_files = glob.glob(os.path.join(transcript_dir,'*_transcription.json'))
  transcript_files.sort()
  rttm_files = glob.glob(os.path.join(transcript_dir,'*.rttm'))
  rttm_files.sort()

  for transcript_file, rttm_file in zip(transcript_files, rttm_files):
    print(transcript_file)
    print(rttm_file)
    assert transcript_file.split('/')[-1][:5] == rttm_file.split('/')[-1][:5] 
    new_outpath = transcript_file.split('transcription.json')[0]+'speaker_turns_diarlm.txt'  ##CHANGE FILENAME HERE 
    new_json_outpath = transcript_file.split('transcription.json')[0]+'speaker_attribution.json'  ##CHANGE FILENAME HERE 
    if os.path.isfile(new_outpath):
      continue
    if 'separate' in new_outpath: #don't process single-speaker recordings
      continue
    transcript = load_transcript(transcript_file)
    diarization = load_rttm(rttm_file)
    speaker_turns = align_words_to_speakers(transcript, diarization)


    flattened_turns = flatten_speaker_turns(speaker_turns)
    speaker_split_list = split_on_spk_tokens(flattened_turns)
    if len(speaker_split_list) == 1: #Skip instances where there is only one speaker
       print('Only one speaker, skipping processing and attribution')
       continue
    window = 30
    hop_length = window
    all_completions = []
    for ii in range(0,len(speaker_split_list),hop_length):
        start_time = time.time()
        HYPOTHESIS = " ".join(speaker_split_list[ii:ii+window]) + " --> "
        print("Tokenizing input...")
        tokens = llm.tokenize(HYPOTHESIS.encode())
        if len(tokens) * 1.2 < 4096: #If the requested tokens fits within the context window
          try:
            print("Decoding completion...")
            completion = llm(HYPOTHESIS, max_tokens=len(tokens) * 1.2)
            print("Transferring completion to hypothesis text...")
            transferred_completion = transfer_llm_completion(completion["choices"][0]["text"], HYPOTHESIS)
            end_time = time.time()
            print("total time taken this loop: ", end_time - start_time)
            all_completions.append(transferred_completion[:-3]) #This strips the --> character at the end of hypothesis
          except:
            HYPOTHESIS = " ".join(speaker_split_list[ii:ii+window]) + " --> "
            transferred_completion = transfer_llm_completion(HYPOTHESIS, HYPOTHESIS)
            end_time = time.time()
            print("total time taken this loop: ", end_time - start_time)
            all_completions.append(transferred_completion[:-3])
        else:
          try:
            small_window = window // 6
            for offset in [0,small_window]:
              HYPOTHESIS = " ".join(speaker_split_list[(ii+offset):(ii+offset+small_window)]) + " --> "
              tokens = llm.tokenize(HYPOTHESIS.encode())
              print("Decoding completion...")
              completion = llm(HYPOTHESIS, max_tokens=len(tokens) * 1.2)
              print("Transferring completion to hypothesis text...")
              transferred_completion = transfer_llm_completion(completion["choices"][0]["text"], HYPOTHESIS)
              end_time = time.time()
              print("total time taken this loop: ", end_time - start_time)
              all_completions.append(transferred_completion[:-3]) #This strips the --> character at the end of hypothesis
          except:
              print('Failure in DiarizationLM decoding, fallback to original Hypothesis')
              HYPOTHESIS = " ".join(speaker_split_list[ii:ii+window]) + " --> "
              transferred_completion = transfer_llm_completion(HYPOTHESIS, HYPOTHESIS)
              end_time = time.time()
              print("total time taken this loop: ", end_time - start_time)
              all_completions.append(transferred_completion[:-3]) #This strips the --> character at the end of hypothesis

    full_completion = ' '.join(all_completions)
    llm_input = create_attribution_prompt(full_completion[:len(full_completion)//4]) #look at the first 25% of the transcript
    attribution_output = new_llm(llm_input, max_tokens=512)
    cleaned = attribution_output['choices'][0]['text'].strip('"` \n"')

    try:
      my_dict = json.loads(cleaned)
      my_dict = {k: ('' if v is None else v) for k, v in my_dict.items()}
      with open(new_json_outpath, 'w', encoding='utf-8') as f:
        json.dump(my_dict, f, ensure_ascii=False, indent=4)
    except:
      print('ERROR IN ATTRIBUTION JSON')
      try:
        my_dict = json_repair.loads(cleaned,skip_json_loads=True)
        my_dict = {k: ('' if v is None else v) for k, v in my_dict.items()}
        with open(new_json_outpath, 'w', encoding='utf-8') as f:
          json.dump(my_dict, f, ensure_ascii=False, indent=4)
      except:
        print('Json Repair failed, default to empty dictionary')
        my_dict = {
          "patient": "",
          "clinician": "",
          "confidence": 0.0,
          "rationale": "",
          "evidence": [
          {"speaker": "", "quote": "", "why": ""},
          {"speaker": "", "quote": "", "why": ""}
          ]
        }
      


    realigned_completion = align_with_dtw(full_completion,speaker_turns)
    my_out = make_aligned_turns_list_postdiarlm(realigned_completion)
    
    with open(new_outpath, 'w') as f:
        for turn in my_out:
            if '> ' in turn:
                speaker, utterance = turn.split(' [')
                # if speaker[1:] == my_dict['patient']:
                if my_dict['patient'] in speaker:
                  f.write(f'Patient [{utterance}\n\n')
                # elif speaker[1:] == my_dict['clinician']:
                elif my_dict['clinician'] in speaker:
                  f.write(f'Clinician [{utterance}\n\n')
                else:
                  f.write(f'{speaker} [{utterance}\n\n')

    print('COMPLETED')


def main():
    parser = argparse.ArgumentParser(description='Transcribe video files using Whisper')
    parser.add_argument('--input_dirs', required=True,
                        help='Input directories containing video files')
    
    args = parser.parse_args()
    
    process_files(
        parent_paths=args.input_dirs
                )


if __name__ == "__main__":
    main()