---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(pl)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Polars

```{index} single: Python; Polars
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade polars yfinance
```

## Overview

[Polars](https://pola.rs/) is a fast data manipulation library for Python written in Rust.

It has gained significant popularity as a modern alternative to {doc}`pandas <pandas>` due to its performance advantages.

Polars is designed with performance and memory efficiency in mind, leveraging:

* [Apache Arrow columnar format](https://arrow.apache.org/docs/format/Columnar.html) for fast data access
* [Lazy evaluation](https://en.wikipedia.org/wiki/Lazy_evaluation) to optimize query execution
* Parallel processing to utilize all available CPU cores
* An expressive API built around column expressions

```{tip}
*Why consider Polars over pandas?*

* **Memory**: pandas typically needs 5--10x your dataset size in RAM; Polars needs only 2--4x
* **Speed**: Polars is 10--100x faster for many common operations
* **See**: [Polars TPC-H benchmarks](https://www.pola.rs/benchmarks/) for up-to-date performance comparisons
```

Throughout the lecture, we will assume that the following imports have taken place

```{code-cell} ipython3
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
```

Like {doc}`pandas`, Polars defines two important data types: `Series` and `DataFrame`.

You can think of a `Series` as a column of data, such as a collection of observations on a single variable.

A `DataFrame` is a two-dimensional object for storing related columns of data.

## Series

```{index} single: Polars; Series
```

Let's start with Series.

We begin by creating a series of four random observations

```{code-cell} ipython3
s = pl.Series(name='daily returns', values=np.random.randn(4))
s
```

```{note}
Unlike {doc}`pandas <pandas>` Series, Polars Series have no row index.
Polars is column-centric --- data access is managed through column expressions
and boolean masks rather than row labels.
See [this blog post](https://medium.com/@luca.basanisi/understand-polars-lack-of-indexes-526ea75e413) for more detail.
```

Polars `Series` are built on top of [Apache Arrow](https://arrow.apache.org/) arrays and support many familiar operations

```{code-cell} ipython3
s * 100
```

Absolute values are available as a method

```{code-cell} ipython3
s.abs()
```

We can also get quick summary statistics

```{code-cell} ipython3
s.describe()
```

Since Polars has no row index, labelled data requires a `DataFrame`.

For example, to associate ticker symbols with returns:

```{code-cell} ipython3
df = pl.DataFrame({
    'company': ['AMZN', 'AAPL', 'MSFT', 'GOOG'],
    'daily returns': np.random.randn(4)
})
df
```

We access a value by filtering on a column expression

```{code-cell} ipython3
df.filter(
    pl.col('company') == 'AMZN'
).select('daily returns').item()
```

Updates also use expressions rather than index assignment

```{code-cell} ipython3
df = df.with_columns(
    pl.when(pl.col('company') == 'AMZN')
    .then(0)
    .otherwise(pl.col('daily returns'))
    .alias('daily returns')
)
df
```

We can also check membership

```{code-cell} ipython3
'AAPL' in df['company']
```

## DataFrames

```{index} single: Polars; DataFrames
```

While a `Series` is a single column of data, a `DataFrame` is several columns, one for each variable.

As in {doc}`pandas`, let's work with data from the [Penn World Tables](https://www.rug.nl/ggdc/productivity/pwt/pwt-releases/pwt-7.0).

We read this in using `pl.read_csv`

```{code-cell} ipython3
url = ('https://raw.githubusercontent.com/QuantEcon/'
       'lecture-python-programming/main/lectures/_static/'
       'lecture_specific/pandas/data/test_pwt.csv')
df = pl.read_csv(url)
df
```

### Selecting data

We can select rows by slicing and columns by name

```{code-cell} ipython3
df[2:5]
```

To select specific columns, pass a list of names to `select`

```{code-cell} ipython3
df.select(['country', 'tcgdp'])
```

These can be combined

```{code-cell} ipython3
df[2:5].select(['country', 'tcgdp'])
```

### Filtering by conditions

The `filter` method accepts boolean expressions built from `pl.col`

```{code-cell} ipython3
df.filter(pl.col('POP') >= 20000)
```

Multiple conditions can be combined with `&` (and) and `|` (or)

```{code-cell} ipython3
df.filter(
    (pl.col('country').is_in(['Argentina', 'India', 'South Africa'])) &
    (pl.col('POP') > 40000)
)
```

Expressions can involve arithmetic across columns

```{code-cell} ipython3
df.filter(
    (pl.col('cc') + pl.col('cg') >= 80) & (pl.col('POP') <= 20000)
)
```

Select the country with the largest household consumption share

```{code-cell} ipython3
df.filter(pl.col('cc') == pl.col('cc').max())
```

### Column expressions

A key difference from pandas is that Polars uses **column expressions** for transformations rather than element-wise `apply` calls.

Here is an example computing the max of each numeric column

```{code-cell} ipython3
df.select(
    pl.col(['year', 'POP', 'XRAT', 'tcgdp', 'cc', 'cg'])
    .max()
    .name.suffix('_max')
)
```

Expressions can be used inside `with_columns` to add or modify columns

```{code-cell} ipython3
df.with_columns(
    (pl.col('XRAT') / 10).alias('XRAT_scaled'),
    pl.col(pl.Float64).round(2)
)
```

Conditional logic uses `pl.when(...).then(...).otherwise(...)`

```{code-cell} ipython3
df.with_columns(
    pl.when(pl.col('POP') >= 20000)
    .then(pl.col('POP'))
    .otherwise(None)
    .alias('POP_filtered')
).select(['country', 'POP', 'POP_filtered'])
```

```{note}
Polars provides `map_elements` as an escape hatch for applying arbitrary
Python functions row-by-row, but it bypasses the optimized expression
engine and should be avoided when a native expression exists.
```

### Missing values

Let's insert some null values to demonstrate imputation techniques

```{code-cell} ipython3
df_nulls = df.with_row_index().with_columns(
    pl.when(pl.col('index') == 0)
    .then(None).otherwise(pl.col('XRAT')).alias('XRAT'),
    pl.when(pl.col('index') == 3)
    .then(None).otherwise(pl.col('cc')).alias('cc'),
    pl.when(pl.col('index') == 5)
    .then(None).otherwise(pl.col('tcgdp')).alias('tcgdp'),
    pl.when(pl.col('index') == 6)
    .then(None).otherwise(pl.col('POP')).alias('POP'),
).drop('index')
df_nulls
```

Fill all nulls with zero

```{code-cell} ipython3
df_nulls.fill_null(0)
```

Or fill with column means

```{code-cell} ipython3
cols = ['cc', 'tcgdp', 'POP', 'XRAT']
df_nulls.with_columns(
    pl.col(cols).fill_null(pl.col(cols).mean())
)
```

Polars also supports forward fill (`fill_null(strategy='forward')`) and interpolation.

There are more [advanced imputation tools](https://scikit-learn.org/stable/modules/impute.html) available in scikit-learn.

### Visualization

Let's build a GDP per capita column and plot it

```{code-cell} ipython3
df = (df
    .select(['country', 'POP', 'tcgdp'])
    .rename({'POP': 'population', 'tcgdp': 'total GDP'})
    .with_columns(
        (pl.col('population') * 1e3).alias('population')
    )
    .with_columns(
        (pl.col('total GDP') * 1e6 / pl.col('population'))
        .alias('GDP percap')
    )
    .sort('GDP percap', descending=True)
)
df
```

We can extract columns directly for matplotlib

```{note}
Polars also provides a built-in [plotting API](https://docs.pola.rs/user-guide/misc/visualization/)
based on Altair (e.g., `df.plot.bar(x=..., y=...)`).
We use matplotlib here for consistency with the rest of the lecture series.
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.bar(df['country'].to_list(), df['GDP percap'].to_list())
ax.set_xlabel('country', fontsize=12)
ax.set_ylabel('GDP per capita', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## Lazy evaluation

```{index} single: Polars; Lazy Evaluation
```

One of Polars' most powerful features is **lazy evaluation**.

Instead of executing each operation immediately, lazy mode collects the full query plan and optimizes it before running.

### Eager vs lazy

```{code-cell} ipython3
# Reload the dataset
url = ('https://raw.githubusercontent.com/QuantEcon/'
       'lecture-python-programming/main/lectures/_static/'
       'lecture_specific/pandas/data/test_pwt.csv')
df_full = pl.read_csv(url)
```

The **eager** API executes immediately (like pandas)

```{code-cell} ipython3
result_eager = (df_full
    .filter(pl.col('tcgdp') > 1000)
    .select(['country', 'year', 'tcgdp'])
    .sort('tcgdp', descending=True)
)
result_eager.head()
```

The **lazy** API builds a query plan instead

```{code-cell} ipython3
lazy_query = (df_full.lazy()
    .filter(pl.col('tcgdp') > 1000)
    .select(['country', 'year', 'tcgdp'])
    .sort('tcgdp', descending=True)
)
print(lazy_query.explain())
```

Call `collect` to execute the plan

```{code-cell} ipython3
result_lazy = lazy_query.collect()
result_lazy.head()
```

### Query optimization

The lazy engine applies several optimizations automatically:

* **Predicate pushdown** --- filters are applied as early as possible
* **Projection pushdown** --- only required columns are read from the source
* **Common subexpression elimination** --- duplicate calculations are merged

Let's see how Polars rewrites a multi-step query

```{code-cell} ipython3
optimized = (df_full.lazy()
    .select(['country', 'year', 'tcgdp', 'POP'])
    .filter(pl.col('tcgdp') > 500)
    .with_columns(
        (pl.col('tcgdp') / pl.col('POP')).alias('gdp_per_capita')
    )
    .filter(pl.col('gdp_per_capita') > 10)
    .select(['country', 'year', 'gdp_per_capita'])
)

print("Optimized plan:")
print(optimized.explain())
```

Executing the plan gives us the final result

```{code-cell} ipython3
optimized.collect()
```

### Performance comparison

Let's compare pandas, Polars eager, and Polars lazy on the same task.

We start with a small dataset (the Penn World Tables we used above) to show
that for small data the differences are negligible

```{code-cell} ipython3
import pandas as pd
import time

# Small dataset -- Penn World Tables (~8 rows)
url = ('https://raw.githubusercontent.com/QuantEcon/'
       'lecture-python-programming/main/lectures/_static/'
       'lecture_specific/pandas/data/test_pwt.csv')
small_pd = pd.read_csv(url)
small_pl = pl.read_csv(url)
```

Now we time the same filter-select-sort operation in each library

```{code-cell} ipython3
# pandas
start = time.perf_counter()
_ = (small_pd
     .query('tcgdp > 500')
     [['country', 'year', 'tcgdp', 'POP']]
     .assign(gdp_pc=lambda d: d['tcgdp'] / d['POP'])
     .sort_values('gdp_pc', ascending=False))
pd_small = time.perf_counter() - start

# Polars eager
start = time.perf_counter()
_ = (small_pl
     .filter(pl.col('tcgdp') > 500)
     .select(['country', 'year', 'tcgdp', 'POP'])
     .with_columns((pl.col('tcgdp') / pl.col('POP')).alias('gdp_pc'))
     .sort('gdp_pc', descending=True))
pl_small = time.perf_counter() - start

print(f"Small data  --  pandas: {pd_small:.4f}s | Polars eager: {pl_small:.4f}s")
```

On a handful of rows the speed difference is immaterial --- use whichever
API you find more convenient.

Now let's scale up to 5 million rows where the difference becomes clear.

The task is: filter rows where `value > 0`, compute a weighted product
`value * weight`, then take the mean of that product within each group ---
a grouped weighted average.

```{code-cell} ipython3
n = 5_000_000
np.random.seed(42)

groups = np.random.choice(['A', 'B', 'C', 'D'], n)
values = np.random.randn(n)
weights = np.random.rand(n)
extra1 = np.random.randn(n)
extra2 = np.random.randn(n)

big_pd = pd.DataFrame({
    'group': groups, 'value': values,
    'weight': weights, 'extra1': extra1, 'extra2': extra2
})
big_pl = pl.DataFrame({
    'group': groups, 'value': values,
    'weight': weights, 'extra1': extra1, 'extra2': extra2
})
```

First, the pandas baseline

```{code-cell} ipython3
start = time.perf_counter()
tmp = big_pd[big_pd['value'] > 0][['group', 'value', 'weight']].copy()
tmp['weighted'] = tmp['value'] * tmp['weight']
_ = tmp.groupby('group')['weighted'].mean()
pd_time = time.perf_counter() - start
print(f"pandas:       {pd_time:.4f}s")
```

Next, Polars in eager mode

```{code-cell} ipython3
start = time.perf_counter()
_ = (big_pl
    .filter(pl.col('value') > 0)
    .select(['group', 'value', 'weight'])
    .with_columns(
        (pl.col('value') * pl.col('weight')).alias('weighted'))
    .group_by('group')
    .agg(pl.col('weighted').mean()))
eager_time = time.perf_counter() - start
print(f"Polars eager: {eager_time:.4f}s")
```

And finally, Polars in lazy mode

```{code-cell} ipython3
start = time.perf_counter()
_ = (big_pl.lazy()
    .filter(pl.col('value') > 0)
    .select(['group', 'value', 'weight'])
    .with_columns(
        (pl.col('value') * pl.col('weight')).alias('weighted'))
    .group_by('group')
    .agg(pl.col('weighted').mean())
    .collect())
lazy_time = time.perf_counter() - start
print(f"Polars lazy:  {lazy_time:.4f}s")
```

The take-away:

* For **small data** (thousands of rows), pandas and Polars perform
  similarly --- choose based on API preference and ecosystem fit.
* For **medium to large data** (hundreds of thousands of rows and above),
  Polars can be significantly faster thanks to its Rust engine, parallel
  execution, and (in lazy mode) query optimization.

The lazy API is particularly powerful when reading from disk --- `scan_csv` returns a `LazyFrame` directly, so filters and projections are pushed down to the file reader.

```{tip}
Use `pl.scan_csv(path)` instead of `pl.read_csv(path)` when working with
large CSV files.
Only the columns and rows you actually need will be read from disk.
See [the Polars I/O documentation](https://docs.pola.rs/user-guide/io/csv/).
```

## On-line data sources

```{index} single: Data Sources
```

As in {doc}`pandas`, Python makes it straightforward to query online databases.

An important database for economists is [FRED](https://fred.stlouisfed.org/) --- a vast collection of time series data maintained by the St. Louis Fed.

Polars' `read_csv` can fetch data from a URL directly.

We use `try_parse_dates=True` to parse the date column automatically

```{code-cell} ipython3
fred_url = ('https://fred.stlouisfed.org/graph/fredgraph.csv?'
            'bgcolor=%23e1e9f0&chart_type=line&drp=0&'
            'fo=open%20sans&graph_bgcolor=%23ffffff&'
            'height=450&mode=fred&recession_bars=on&'
            'txtcolor=%23444444&ts=12&tts=12&width=1318&'
            'nt=0&thu=0&trc=0&show_legend=yes&'
            'show_axis_titles=yes&show_tooltip=yes&'
            'id=UNRATE&scale=left&cosd=1948-01-01&'
            'coed=2024-06-01&line_color=%234572a7&'
            'link_values=false&line_style=solid&'
            'mark_type=none&mw=3&lw=2&ost=-99999&'
            'oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&'
            'fgst=lin&fgsnd=2020-02-01&line_index=1&'
            'transformation=lin&vintage_date=2024-07-29&'
            'revision_date=2024-07-29&nd=1948-01-01')
data = pl.read_csv(fred_url, try_parse_dates=True)
```

Let's inspect the first few rows

```{code-cell} ipython3
data.head()
```

And get summary statistics

```{code-cell} ipython3
data.describe()
```

Plot the unemployment rate from 2006 to 2012

```{code-cell} ipython3
filtered = data.filter(
    (pl.col('observation_date') >= pl.date(2006, 1, 1)) &
    (pl.col('observation_date') <= pl.date(2012, 12, 31))
)

fig, ax = plt.subplots()
ax.plot(filtered['observation_date'].to_list(),
        filtered['UNRATE'].to_list())
ax.set_title('US Unemployment Rate')
ax.set_xlabel('year', fontsize=12)
ax.set_ylabel('%', fontsize=12)
plt.show()
```

Polars supports [many file formats](https://docs.pola.rs/user-guide/io/) including Excel, JSON, Parquet, and direct database connections.

## Exercises

```{exercise-start}
:label: pl_ex1
```

With these imports:

```{code-cell} ipython3
import datetime as dt
import yfinance as yf
```

Write a program to calculate the percentage price change over 2021 for the following shares:

```{code-cell} ipython3
ticker_list = {'INTC': 'Intel',
               'MSFT': 'Microsoft',
               'IBM': 'IBM',
               'BHP': 'BHP',
               'TM': 'Toyota',
               'AAPL': 'Apple',
               'AMZN': 'Amazon',
               'C': 'Citigroup',
               'QCOM': 'Qualcomm',
               'KO': 'Coca-Cola',
               'GOOG': 'Google'}
```

Here's a function that reads closing prices into a Polars DataFrame:

```{code-cell} ipython3
def read_data_polars(ticker_list,
                     start=dt.datetime(2021, 1, 1),
                     end=dt.datetime(2021, 12, 31)):
    """
    Read closing price data from Yahoo Finance
    and return a Polars DataFrame.
    """
    dataframes = []

    for tick in ticker_list:
        stock = yf.Ticker(tick)
        prices = stock.history(start=start, end=end)
        df = pl.DataFrame({
            'Date': list(prices.index.date),
            tick: prices['Close'].values
        }).with_columns(pl.col('Date').cast(pl.Date))
        dataframes.append(df)

    result = dataframes[0]
    for df in dataframes[1:]:
        result = result.join(
            df, on='Date', how='full', coalesce=True
        )
    return result

ticker = read_data_polars(ticker_list)
```

Complete the program to plot the result as a bar graph.

```{exercise-end}
```

```{solution-start} pl_ex1
:class: dropdown
```

Calculate percentage changes using Polars expressions:

```{code-cell} ipython3
price_change = ticker.select([
    ((pl.col(tick).last() / pl.col(tick).first() - 1) * 100)
    .alias(tick)
    for tick in ticker_list.keys()
]).transpose(
    include_header=True,
    header_name='ticker',
    column_names=['pct_change']
).with_columns(
    pl.col('ticker')
    .replace_strict(ticker_list, default=pl.col('ticker'))
    .alias('company')
).sort('pct_change')

print(price_change)
```

Plot the results using matplotlib directly:

```{code-cell} ipython3
companies = price_change['company'].to_list()
changes = price_change['pct_change'].to_list()
colors = ['red' if x < 0 else 'blue' for x in changes]

fig, ax = plt.subplots(figsize=(10, 8))
ax.bar(companies, changes, color=colors)
ax.set_xlabel('stock', fontsize=12)
ax.set_ylabel('percentage change in price', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

```{solution-end}
```


```{exercise-start}
:label: pl_ex2
```

Using `read_data_polars` from {ref}`pl_ex1`, obtain year-on-year percentage change for these indices:

```{code-cell} ipython3
indices_list = {'^GSPC': 'S&P 500',
               '^IXIC': 'NASDAQ',
               '^DJI': 'Dow Jones',
               '^N225': 'Nikkei'}
```

Plot the result as a time series graph.

```{exercise-end}
```

```{solution-start} pl_ex2
:class: dropdown
```

```{code-cell} ipython3
indices_data = read_data_polars(
    indices_list,
    start=dt.datetime(1971, 1, 1),
    end=dt.datetime(2021, 12, 31)
)

indices_data = indices_data.with_columns(
    pl.col('Date').dt.year().alias('year')
)
```

Calculate yearly returns using group-by operations:

```{code-cell} ipython3
yearly_returns = indices_data.group_by('year').agg([
    *[pl.col(idx).drop_nulls().first().alias(f'{idx}_first')
      for idx in indices_list],
    *[pl.col(idx).drop_nulls().last().alias(f'{idx}_last')
      for idx in indices_list]
])

for idx, name in indices_list.items():
    yearly_returns = yearly_returns.with_columns(
        ((pl.col(f'{idx}_last') - pl.col(f'{idx}_first'))
         / pl.col(f'{idx}_first') * 100).alias(name)
    )

yearly_returns = (yearly_returns
    .select(['year', *indices_list.values()])
    .sort('year')
)
print(yearly_returns)
```

Summary statistics:

```{code-cell} ipython3
yearly_returns.select(list(indices_list.values())).describe()
```

Plot each index in a subplot:

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
years = yearly_returns['year'].to_list()

for iter_, ax in enumerate(axes.flatten()):
    name = list(indices_list.values())[iter_]
    values = yearly_returns[name].to_list()
    ax.plot(years, values, 'o-', linewidth=2, markersize=4)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('yearly return (%)', fontsize=12)
    ax.set_xlabel('year', fontsize=12)
    ax.set_title(name, fontsize=12)

plt.tight_layout()
plt.show()
```

```{solution-end}
```

[^mung]: Wikipedia defines munging as cleaning data from one raw form into a structured, purged one.
