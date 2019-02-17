# -*- coding: utf-8 -*-
"""
This material is from the internet, use only for study.

"""

# https://www.youtube.com/watch?v=suRd3UzdBeo
# code :  https://github.com/llSourcell/kaggle_challenge

# exp: write comments to guide youself

# log of the target value
train[""] = np.log(train[""].values + 1)
plt.hist(train[""], bins=100)
plt.xlabel("")
plt.ylabel("")
plt.show()



xgb.train("default", train)