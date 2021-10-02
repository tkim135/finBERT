from tokenizers import Tokenizer, Encoding
import pathlib
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union
)

class BaseTokenizer(object):
    """Inherit from this ABC to build a CT2 tokenizer.
    """

    def __call__(self, text: Union[str, List[str]], *args, **kwargs) -> Dict[str, Any]:
        pass

    def __len__(self) -> int:
        """Vocab size.
        """
        pass

    def md5(self) -> str:
        """Hash summarising configuration of an instance.
        """

class GPT2Tokenizer(BaseTokenizer):

    def __init__(self,
                 vocab_file : str,
                 name: Union[str, pathlib.Path] = 'gpt',
                 version: Optional[str] = '2.0.0',
                 unk_token: str = '<|unktoken|>',
                 eos_token: str = '\n\n',
                 pad_token: Optional[str] = '<|padtoken|>',  # If None do not pad.
                 pad_direction: str = 'right',
                 pad_to_length: int = 48,
                 ignore_cached: bool = False):

        path_to_tokenizer = pathlib.Path(vocab_file) # / f"{name}.{version}.json"
        self._tokenizer = Tokenizer.from_file(path_to_tokenizer.as_posix())

        self._name = name
        self._version = version
        self._unk_token = unk_token
        self._eos_token = eos_token
        self._pad_token = pad_token

        self._pad_direction = pad_direction
        self._pad_to_length = pad_to_length

        special_tokens = [unk_token, eos_token] 
        if pad_token is not None:
            special_tokens.append(pad_token)
        self.add_tokens(special_tokens, special_tokens=True)

        self._configure_padding()
        self._tokenizer.no_truncation()

    def __call__(self, text: Union[str, List[str]]) -> List[Encoding]:

        if isinstance(text, str):
            text = [text]

        # Add eos tokens.
        text = [el + self._eos_token for el in text]

        return self._tokenizer.encode_batch(text, add_special_tokens=True, is_pretokenized=False)

    def __len__(self) -> int:
        return self.vocab_size

    def __str__(self) -> str:
        str_repr = f"""
            {self.__class__.__name__}
             name:{self._name}
             version:{self._version}
             unk_token:{self._unk_token}
             eos_token:{self._eos_token}
             pad_token:{self._pad_token}
             pad_direction:{self._pad_direction}
             pad_to_length:{self._pad_to_length}
        """
        return str_repr

    @property
    def md5(self) -> str:
        encoder = json.dumps(self.vocab, sort_keys=True)
        config = f"{str(self)}:{encoder}"
        return hashlib.md5(config.encode('utf-8')).hexdigest()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def pad_token_id(self) -> Optional[int]:
        if self._pad_token is None:
            return None
        return self.token_to_id(self._pad_token)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id(self._eos_token)

    def add_tokens(self, tokens: List[str], special_tokens: bool = False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(tokens)
        return self._tokenizer.add_tokens(tokens)

    def token_to_id(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            index = self._tokenizer.token_to_id(self._unk_token)
        return index

    def id_to_token(self, idx: int) -> str:
        return self._tokenizer.id_to_token(idx)
   
    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens
    def convert_tokens_to_ids(self, input_tokens):
        ids = []
        for tokens in input_tokens:
            ids.append(self.token_to_id(tokens))
        return ids
    def _configure_padding(self):
        if not self._pad_token:
            self._tokenizer.no_padding()
            return

        self._tokenizer.enable_padding(
            length=None,
            direction=self._pad_direction,
            pad_id=self.pad_token_id,
            pad_type_id=0,
            pad_token=self._pad_token,
            pad_to_multiple_of=self._pad_to_length
        )
