################################################################################
# Sven Boogmans
# 16-09-2020
# Fundementals of Datascience assignment 1
################################################################################ now-cast U.S.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import scipy

def main():

    # check_correlation()

    # plot_nowcast_polls()
    # plot_pos_neg_tweets_trump()
    # plot_pos_neg_tweets_clinton()


def check_correlation():
    poll_dataframe = pd.read_pickle("poll_dataframe.pkl")
    poll_dataframe.enddate = pd.to_datetime(poll_dataframe.enddate).dt.date
    poll_dataframe = poll_dataframe.loc[poll_dataframe["type"] == "now-cast"]
    poll_dataframe = poll_dataframe.loc[poll_dataframe["state"] == "U.S."]
    poll_dataframe = poll_dataframe.loc[poll_dataframe["grade"].isin(["A", "A+", "A-", "B", "B+", "B-", "C", "C+", "C-"])]

    mean_trump_dataframe = pd.DataFrame(columns=["mean_trump"])
    for enddate in poll_dataframe.enddate.unique():
         mean_poll_on_trump = poll_dataframe["adjpoll_trump"].loc[poll_dataframe["enddate"] == enddate].mean()
         mean_trump_dataframe.loc[enddate,"mean_trump"] = mean_poll_on_trump/100

    mean_trump_dataframe = mean_trump_dataframe.dropna()
    mean_trump_dataframe = pd.Series(mean_trump_dataframe["mean_trump"], index=mean_trump_dataframe.index)
    mean_trump_dataframe = pd.to_numeric(mean_trump_dataframe, errors='coerce')


    mean_hillary_dataframe = pd.DataFrame(columns=["mean_hillary"])
    for enddate in poll_dataframe.enddate.unique():
         mean_poll_on_hillary = poll_dataframe["adjpoll_clinton"].loc[poll_dataframe["enddate"] == enddate].mean()
         mean_hillary_dataframe.loc[enddate,"mean_hillary"] = mean_poll_on_hillary/100

    mean_hillary_dataframe = mean_hillary_dataframe.dropna()
    mean_hillary_dataframe = pd.Series(mean_hillary_dataframe["mean_hillary"], index=mean_hillary_dataframe.index)
    mean_hillary_dataframe = pd.to_numeric(mean_hillary_dataframe, errors='coerce')

    tweet_dataframe = pd.read_pickle("tweet_dataframe.pkl")
    tweet_dataframe.date = pd.to_datetime(tweet_dataframe.date)#.dt.date
    hillary_tweets = tweet_dataframe[tweet_dataframe["text"].str.contains("@HillaryClinton")]
    trump_tweets = tweet_dataframe[tweet_dataframe["text"].str.contains("@realDonaldTrump")]

    hillary_positive = hillary_tweets.loc[hillary_tweets.sentiment=="Positive", "date"]
    hillary_negative = hillary_tweets.loc[hillary_tweets.sentiment=="Negative", "date"]

    trump_positive = trump_tweets.loc[trump_tweets.sentiment=="Positive", "date"]
    trump_negative = trump_tweets.loc[trump_tweets.sentiment=="Negative", "date"]

    positive_hillary_counts = hillary_positive.dt.date.value_counts()
    negative_hillary_counts = hillary_negative.dt.date.value_counts()
    neg_hillary_proportion = negative_hillary_counts.divide(positive_hillary_counts.add(negative_hillary_counts))
    neg_hillary_proportion = neg_hillary_proportion.rename("neg_clinton_prop")
    pos_hillary_proportion = positive_hillary_counts.divide(positive_hillary_counts.add(negative_hillary_counts))
    pos_hillary_proportion = pos_hillary_proportion.rename("pos_clinton_prop")

    positive_trump_counts = trump_positive.dt.date.value_counts()
    negative_trump_counts = trump_negative.dt.date.value_counts()
    neg_trump_proportion = negative_trump_counts.divide(positive_trump_counts.add(negative_trump_counts))
    neg_trump_proportion = neg_trump_proportion.rename("neg_trump_prop")
    pos_trump_proportion = positive_trump_counts.divide(positive_trump_counts.add(negative_trump_counts))
    pos_trump_proportion = pos_trump_proportion.rename("pos_trump_prop")

    all_data = pd.concat([pos_trump_proportion,
                          neg_trump_proportion,
                          pos_hillary_proportion,
                          neg_hillary_proportion,
                          mean_trump_dataframe,
                          mean_hillary_dataframe], axis=1)

    all_data = all_data.dropna()

    corr, pvalue = scipy.stats.kendalltau(all_data["neg_trump_prop"].tolist(), all_data["pos_clinton_prop"].tolist())
    print("corr; ", corr)
    print("pvalue; ", pvalue)

        #print(neg_trump_proportion.head(), pos_trump_proportion.head(), neg_hillary_proportion.head(), pos_hillary_proportion.head())

# function to plot now cast poll data
def plot_nowcast_polls():
    poll_dataframe = pd.read_pickle("poll_dataframe.pkl")
    poll_dataframe = poll_dataframe.loc[poll_dataframe["type"] == "now-cast"]
    poll_dataframe = poll_dataframe.loc[poll_dataframe["state"] == "U.S."]
    poll_dataframe = poll_dataframe.loc[poll_dataframe["grade"].isin(["A", "A+", "A-", "B", "B+", "B-", "C", "C+", "C-"])]
    poll_dataframe["ordinal_dates"] = poll_dataframe.enddate.apply(lambda x: x.to_pydatetime().toordinal())

    x_ordinal_dates = poll_dataframe.ordinal_dates
    x_normal_dates = poll_dataframe.enddate

    plt.scatter(x_normal_dates, poll_dataframe.adjpoll_clinton, alpha=0.5, color="b", label="Would vote for Clinton")
    z1 = np.polyfit(x_ordinal_dates, poll_dataframe.adjpoll_clinton, 1)
    p1 = np.poly1d(z1)
    plt.plot(x_normal_dates,p1(x_ordinal_dates), alpha=0.5, color = "b")

    plt.scatter(x_normal_dates, poll_dataframe.adjpoll_trump, alpha=0.5, color="r", label="Would vote for Trump")
    z2 = np.polyfit(x_ordinal_dates, poll_dataframe.adjpoll_trump, 1)
    p2 = np.poly1d(z2)
    plt.plot(x_normal_dates,p2(x_ordinal_dates), alpha=0.5, color = "r")

    plt.xticks(rotation=45)
    plt.legend()
    plt.ylabel("% Votes")
    plt.xlabel("Last day of the poll")
    plt.title("Votes on Trump vs Clinton according to 88 now-cast polls")
    plt.tight_layout()
    plt.savefig("now_cast")

# function to plot negative and positive data on Clinton
def plot_pos_neg_tweets_clinton():
    tweet_dataframe = pd.read_pickle("tweet_dataframe.pkl")
    tweet_dataframe["date"] = pd.to_datetime(tweet_dataframe.date).dt.date
    tweet_dataframe["ordinal_dates"] = tweet_dataframe.date.apply(lambda x: x.toordinal())
    hillary_tweets = tweet_dataframe[tweet_dataframe["text"].str.contains("@HillaryClinton")]

    x1 = hillary_tweets.loc[hillary_tweets.sentiment=="Positive", "ordinal_dates"]
    x2 = hillary_tweets.loc[hillary_tweets.sentiment=="Negative", "ordinal_dates"]
    x3 = hillary_tweets.loc[hillary_tweets.sentiment=="Positive", "date"]
    x4 = hillary_tweets.loc[hillary_tweets.sentiment=="Negative", "date"]

    kwargs = dict(alpha=0.5, bins=32)
    fig, ax = plt.subplots(2)
    (n1, bins, patches) = ax[0].hist(x3, **kwargs, color="g", label="Positive")
    (n2, bins, patches) = ax[0].hist(x4, **kwargs, color="r", label="Negative")
    ax[0].legend()
    ax[0].set_xticklabels([])
    ax[0].set_ylabel("Frequency")

    x_ordinal_dates = sorted(x1.unique())
    x_normal_dates = sorted(x3.unique())
    ax[1].scatter(x_normal_dates, n1/(n1+n2), alpha=0.5, color="g", label="Positive")
    ax[1].scatter(x_normal_dates, n2/(n1+n2), alpha=0.5, color="r", label="Negative")

    z1 = np.polyfit(x_ordinal_dates, n2/(n1+n2), 1)
    p1 = np.poly1d(z1)
    ax[1].plot(x_normal_dates,p1(x_ordinal_dates),"r--", alpha=0.5, color = "r")

    z2 = np.polyfit(x_ordinal_dates, n1/(n1+n2), 1)
    p2 = np.poly1d(z2)
    ax[1].plot(x_normal_dates,p2(x_ordinal_dates),"r--", alpha=0.5, color = "g")

    ax[1].set_ylabel("Proportion")
    ax[1].set_ylim(0.2,0.8)
    ax[1].tick_params(labelrotation=45)
    fig.suptitle("Positive and negative tweets that mention @HillaryClinton")
    fig.tight_layout()
    fig.savefig("Clinton_pos_neg")


# function to plot negative and positive data on Trump
def plot_pos_neg_tweets_trump():
    tweet_dataframe = pd.read_pickle("tweet_dataframe.pkl")
    tweet_dataframe["date"] = pd.to_datetime(tweet_dataframe.date).dt.date
    tweet_dataframe["ordinal_dates"] = tweet_dataframe.date.apply(lambda x: x.toordinal())
    trump_tweets = tweet_dataframe[tweet_dataframe["text"].str.contains("@realDonaldTrump")]

    x1 = trump_tweets.loc[trump_tweets.sentiment=="Positive", "ordinal_dates"]
    x2 = trump_tweets.loc[trump_tweets.sentiment=="Negative", "ordinal_dates"]
    x3 = trump_tweets.loc[trump_tweets.sentiment=="Positive", "date"]
    x4 = trump_tweets.loc[trump_tweets.sentiment=="Negative", "date"]

    kwargs = dict(alpha=0.5, bins=32)
    fig, ax = plt.subplots(2)
    (n1, bins, patches) = ax[0].hist(x3, **kwargs, color="g", label="Positive")
    (n2, bins, patches) = ax[0].hist(x4, **kwargs, color="r", label="Negative")
    ax[0].legend()
    ax[0].set_xticklabels([])
    ax[0].set_ylabel("Frequency")

    x_ordinal_dates = sorted(x1.unique())
    x_normal_dates = sorted(x3.unique())
    ax[1].scatter(x_normal_dates, n1/(n1+n2), alpha=0.5, color="g", label="Positive")
    ax[1].scatter(x_normal_dates, n2/(n1+n2), alpha=0.5, color="r", label="Negative")

    z1 = np.polyfit(x_ordinal_dates, n2/(n1+n2), 1)
    p1 = np.poly1d(z1)
    ax[1].plot(x_normal_dates,p1(x_ordinal_dates),"r--", alpha=0.5, color = "r")

    z2 = np.polyfit(x_ordinal_dates, n1/(n1+n2), 1)
    p2 = np.poly1d(z2)
    ax[1].plot(x_normal_dates,p2(x_ordinal_dates),"r--", alpha=0.5, color = "g")

    ax[1].set_ylabel("Proportion")
    ax[1].set_ylim(0.2,0.8)
    ax[1].tick_params(labelrotation=45)
    fig.suptitle("Positive and negative tweets that mention @realDonaldTrump")
    fig.tight_layout()
    fig.savefig("Trump_pos_neg")


if __name__ == "__main__":
    main()
