
from  dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

def plot(years, petaflops, labels, label_pos, color, line_color="go-"):
    years = [parser.parse(year) for year in years]

    years, petaflops = np.array(years), np.array(petaflops)

    ax.scatter(years, petaflops, c=color)
    for i, _ in enumerate(years):
        ax.text(years[i], petaflops[i]+label_pos[i], labels[i])
        # ax.annotate(labels[i], (years[i], petaflops[i]))

    plt.plot([years[0], years[-1]], [petaflops[0], petaflops[-1]], line_color)


not_transformers = ["AlexNet", "Seq2Seq", "VGG-19", "Resnet", "InceptionV3", "Xception", "ResNeXt", "DenseNet201", "ELMo", "MoCo ResNet50", "Wav2Vec 2.0",]
nt_years = ["30 Sep 2012", "10 Sep 2014", "4 Sep 2014", "10 Dec 2015", "2 Dec 2015", "7 Oct 2016", "16 Nov 2016", "25 Aug 2016", "15 Feb 2018", "13 Nov 2019", "20 Jun 2020"]
nt_petaflops = [2.7, 4, 4.1, 3.95, 4.97, 5.65, 4.15, 3.45, 3.55, 5.96, 6.23]
nt_text_pos = [-.15, -.15, .05, .05, .05, .05, .05, -.15, .05, .05, .05]

transformers = ["Transformer", "GPT-1", "BERT Large", "Switch \nTransformer \n1.6T", "XLNet", "Megatron", "GPT-2", "Microsoft T-NLG", "GPT-3", "Megatron-Turing NLG 530B"]
t_years = ["12 Jun 2017", "June 11, 2018", "11 Oct 2018", "11 Jan 2021", "19 Jun 2019", "17 Sep 2019", "14 Feb 2019", "13 Feb 2020", "28 May 2020", "28 Jan 2022"]
t_petaflops = [ 4.4, 4.75, 5.4, 5.76, 6.2, 6.9, 7.3, 7.45, 8.5, 9]
t_text_pos = [.05, .05, .05, -.5, .05, .05, .05, .05, .05, .05, .05]

# plt.figure()
fig, ax = plt.subplots()
fig.set_figheight(18)
fig.set_figwidth(8)

plot(nt_years, nt_petaflops, not_transformers, nt_text_pos, "green", "go-")
plot(t_years, t_petaflops, transformers, t_text_pos, "red", "ro-")

yticklabels = [pow(10, i) for i in range(2, 10)]
yticks = np.array([math.log(yticklabel, 10) for yticklabel in yticklabels])
print(yticks)

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)


plt.show()
# plt.savefig("image_name.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)
