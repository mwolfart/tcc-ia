import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    top_words = [("fight", 6.036170), ("privaci", 6.232774), ("►", 6.254600), ("the", 6.327015), ("term", 6.379058), ("autorenew", 6.413746), ("collect", 6.427839), ("battl", 6.443762), ("player", 6.601092), ("user", 6.722742), ("turn", 6.771946), ("casino", 7.089014), ("win", 7.103004), ("get", 7.142892), ("current", 7.833625), ("trial", 7.967947), ("end", 8.182977), ("charg", 8.785523), ("may", 9.463728), ("renew", 9.759226), ("●", 9.820934), ("free", 9.965301), ("account", 10.750569), ("purchas", 11.025802), ("game", 11.452922), ("period", 12.424583), ("slot", 12.482132), ("play", 12.543214), ("•", 21.671877), ("subscript", 24.830651)]
    
    good_rating_stats_svm = [("4.5", 77.31, 77.48, 97.05), ("4.6", 67.97, 68.39, 76.50), ("4.7", 73.69, 67.95, 35.60)]
    good_rating_stats_hgb = [("4.5", 78.82, 80.51, 93.35), ("4.6", 69.14, 71.24, 74.25)]
    
    high_eng_stats_svm = [("20.000", 71.92, 72.64, 79.64), ("30.000", 71.16, 70.22, 66.95), ("50.000", 73.28, 70.71, 52.09)]
    high_eng_stats_hgb = [("20.000", 72.87, 74.53, 78.92), ("30.000", 74.07, 73.16, 71.35)]
    
    df = pd.DataFrame(high_eng_stats_hgb)
    df = df.rename(columns={0: "Breakpoint", 1: "Accuracy", 2: "Precision", 3: "Recall"})
    
    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    sns.lineplot(df, x="Breakpoint", y="Accuracy", label="Accuracy")
    sns.lineplot(df, x="Breakpoint", y="Precision", label="Precision")
    sns.lineplot(df, x="Breakpoint", y="Recall", label="Recall")
    plt.show()
    return

if __name__ == "__main__":
    main()