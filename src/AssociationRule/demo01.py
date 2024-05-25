# 加载包。这里使用efficient-apriori，python中也可以利用apyori库和mlxtend库
from efficient_apriori import apriori

"""
apriori(transactions: typing.Iterable[typing.Union[set, tuple, list]],
        min_support: float=0.5,
        min_confidence: float=0.5,
        max_length: int=8,
        verbosity: int=0,
        output_transaction_ids: bool=False)
上面就是这个函数的参数
min_support：最小支持度
min_confidence：最小置信度
max_length：项集长度
"""

