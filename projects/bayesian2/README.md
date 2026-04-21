# Control Charting with Bayesian Modeling

> WIP: leaving some notes while this is being worked on.

This is a extension of the Bayesian Modeling of a Lab Assay. The example here extends use of the
code to show trends over time. Let's say that something interesting happens over time. Consider these examples that are all based
off of things I've actually observed previously:

1. Runs are bad then get better. It turns out tech "Bob" was learning the assay and was negatively
   Impacting results for the first month he ran it. In the second month, Bob's performance matche
   Jane, the more senior tech.
2. The instrument clogs over time. Normal maintenance fixes it; however, annoyingly around
   1 out of 10 runs has issues.
3. Instrument "B" has a shaker motor that burned out near the end of month 1. It needs repair and is
   notably worse in month two.

All of the above are happening at the same time; however, applying the model to a rolling window of
the data will make results that let us tease out each of the three things above. Without modeling,
it is likely that everyone on the team will either make up guesses about what is going on or
otherwise just assume that it is all just normal variations that happen.

Modeling gives a powerful tool to quickly react to things that the lab can proactively do: (re)train
techs that consistently have sub-par results and call vendors for instrument maintenance as soon as
possible.

Below shows example data that we'll create. Assume one run of 10 plates is done each week. There are
8 runs of data generated below.

```python
# TODO: params for runs
```

The model for these runs captures run (a proxy for time), tech, reagent and instrument. The trick
here is that we'll split data to show month 1 vs month 2. Something like a monthly report.

```python
#TODO: show the model and code to group it by month
```

Above could also be fancier and use a rolling window that models the last X runs of data each time
a new run is done. Plotting it would show changes more granularly; however, it is not done here in
order to keep this example relatively simple.

```bash
# TODO running the example

# TODO: model output
```
Figures are also generated showing month 1 versus month 2. Notice ...

TODO: table with Month 1 on left and month 2 on right.

Data from the model's output can also be plotted over time. For example, this might be a high
level control chart so that you don't need to look at a bunch of individual plots (like above)

TODO: show chart with a rolling window of the last 4 runs for the two months

# Conclusion

Bayesian modeling is a powerful tool for letting the data tell you what is happening. It is easy to
do in Python!

Any sort of high-throughput process, including lab workflows should use something similar to what
I've shown here. From experience, it isn't uncommon that most labs can make a lot of data but don't
have a good strategy for how to explain what is causing good or bad performance. Instrument issues,
tech performance and reagent lots are all real examples that are straight-forward to model.