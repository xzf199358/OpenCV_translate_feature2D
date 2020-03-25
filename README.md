# OpenCV_translate_feature2D
this repository will show the  translation of opencv official word  that including code in three languages c++ python and java
# Harris 角点检测
在本教程中，您将学习：</br>
有哪些特征及其重要性。</br>
使用函数cv：：cornerHarris使用Harris-Stephens方法检测角点。</br>
## 理论
### 什么是特征？
&emsp;在计算机视觉中，通常需要在环境的不同帧之间找到匹配点。为什么？如果我们知道两个图像是如何相互关联的，我们可以使用这两个图像来提取它们的信息。</br>
&emsp;当我们说匹配点时，一般来说，我们指的是场景中容易识别的特征。我们称这些特征为特征。</br>
&emsp;那么，一个特征应该具有哪些特征呢？它必须是唯一可识别的。</br>
## 图像特征类型
举几个例子：</br>
&emsp;边缘</br>
&emsp;角点（也称为兴趣点）</br>
&emsp;Blobs（也称为感兴趣的区域）</br>
在本教程中，我们将特别研究角点特征。
## 为什么角点这么特别？
&emsp;因为它是两条边的交集,它表示这两条边的方向改变的点。因此，图像的梯度（在两个方向上）有很大的变化，可以用来检测它。
## 它是如何工作的？
我们寻找一个角点，由于角点代表图像中渐变的变化，我们将寻找这种“变化”。</br>
考虑一个*灰度图像*$\(I)$。我们将扫描一个窗口$\w（x，y）$,（位移u在x方向，v在y方向）$\I$并计算强度的变化。</br>
质能守恒方程 $\pi$ 可以用一个很简洁的方程式 $E=mc^2$ 来表达。<br/>
#  $$E=mc^2$$

# 【读书笔记】概率论与数理统计（上）

作者：LogM

本文原载于 [https://segmentfault.com/u/logm/articles](https://segmentfault.com/u/logm/articles)，不允许转载~

文章中的数学公式若无法正确显示，请参见：[正确显示数学公式的小技巧](https://segmentfault.com/a/1190000019359797)

本文为[概率论与数理统计](https://www.icourse163.org/learn/ZJU-232005)的笔记。

------

## 1. 第一周

- ### 1.1 样本空间

  - 随机试验的所有可能$S = \{e\}$

- ### 1.2 事件

  - 和事件：$A \cup B$

  - 积事件：$A \cap B$、$AB$

  - 差事件：$A-B$
  
  - 对立事件：$\overline{A}$

- ### 1.3 常用公式

  - $\overline{A} \cap \overline{B} = \overline{A \cup B}$、$\overline{A} \cup \overline{B} = \overline{A \cup B}$

  - $P(A\overline{B}) = P(A-B) = P(A)-P(AB)$

  - 加法公式：$P(A \cup B) = P(A) + P(B) - P(AB)$、$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(AB) - P(BC) - P(AC) + P(ABC)$

------

## 2. 第二周

- ### 2.1 古典概型（等可能概型）

  - 抽球问题：N个球，其中a个白球，b个黄球，不放回抽n次，球恰好k个白球概率
  $$
  P = \frac{C_a^k \cdot C_b^{n-k}}{C_{a+b}^n}
  $$

  - 生日问题：n个人，至少2人同生日的概率
  $$
  P = 1 - \frac{A_{365}^n}{365^n}
  $$

  - 抽签问题：a个白球，b个黄球，不放回抽n次，第k次为白球的概率
  $$
  P = \frac{a \cdot (a+b-1)!}{(a+b)!} = \frac{a}{a+b}
  $$

  - > 不放回抽样，第k次抽到白球的概率等于第一次抽到白球的概率

- ### 2.2 条件概率

  - $P(B|A) = \frac{P(AB)}{P(A)}$
  
  - $P(ABC) = P(A) \cdot P(B|A) \cdot P(C|AB)$

- ### 2.3 全概率公式

- ### 2.4 贝叶斯公式

------

## 3. 第三周

- ### 3.1 0-1分布（两点分布、贝努利分布）

  - 记为：$X \sim 0-1(p)$、$X \sim B(p)$

  - 分布律：$P(X=k) = (1-p)^{1-k} \cdot p^k$

  - 期望：$E(X) = p$

  - 方差：$D(X) = E(X^2)-[E(X)]^2 = p-p^2$

- ### 3.2 二项分布（Binomial）

  - 记为：$X \sim B(n, p)$
  
  - 分布律：$P(X=k) = C_n^k \cdot p^k \cdot (1-p)^{n-k}$

  - 期望：$E(X) = np$

  - 方差：$E(X) = n(p-p^2)$

  - 若 $X \sim B(n_1, p)$，$Y \sim B(n_2, p)$，并且 $X$ 与 $Y$ 相互独立，则$Z=X+Y \sim B(n_1+n_2, p)$

- ### 3.3 泊松分布（Poisson）

  - 记为：$X \sim \pi(\lambda)$、$X \sim P(\lambda)$

  - 分布律：$P(X=k) = \frac{\lambda^k \cdot e^{-\lambda}}{k!}$

  - 期望：$E(X) = \lambda$

  - 方差：$D(X) = E(X^2)-[E(X)]^2 = (\lambda^2+\lambda) - \lambda^2$
  
  - > 事件以固定强度$\lambda$随机独立发生，则该事件在单位时间发生次数k可视为泊松分布。强度$\lambda$可由单位时间该事件发生次数的平均数统计得到。

  - > 泊松分布的本质：当二项分布的 $n>10$、$p<0.1$、$\lambda=np$ 时，泊松分布近似二项分布。

  - 若 $X \sim \pi(\lambda_1)$，$Y \sim \pi(\lambda_2)$，并且 $X$ 与 $Y$ 相互独立，则$Z=X+Y \sim \pi(\lambda_1 + \lambda_2)$

- ### 3.4 几何分布（Geometric）

  - 记为：$X \sim Geom(p)$
  
  - 分布律：$P(X=k) = p \cdot (1-p)^{k-1}$

  - 期望：$E(X) = 1/p$

  - > 掷骰子，掷到6点就停止，掷骰子的次数k服从几何分布。

- ### 3.5 概率分布函数

  - $F_X(x) = P(X \leq x)$

------

## 4. 第四周

- ### 4.1 概率密度函数

  - 概率分布函数$F_X(x)$，概率密度函数$f_X(x)$

  - 连续型随机变量满足：$F(x) = \int_{-\infty}^{x} f(t)dt$、$F^{'}(x)=f(x)$

  - 连续型随机变量单点取值的概率为0：$P(x=a) = 0$

- ### 4.2 均匀分布（Uniform）

  - 记为：$X \sim U(a, b)$、$X \sim Unif(a, b)$

  - 期望：$E(X) = \frac{a+b}{2}$

  - 方差：$D(X) = E(X^2)-[E(X)]^2 = \frac{a^2+b^2+ab}{3} - [\frac{a+b}{2}]^2$

  - 若 $X$ 与 $Y$ 相互独立，且 $X \sim U(0, 1)$，$Y \sim U(0, 1)$，则 $Z=X+Y$ 的概率密度为：$f_Z(z)=\left \{\begin{matrix} z & ,0 \leq z \leq 1 \\ 2-z & ,1 < z \leq 2  \\ 0  & ,others \end {matrix} \right.$

- ### 4.3 指数分布（Exponential）

  - 记为：$X \sim E(\lambda)$、$X \sim Exp(\lambda)$
  
  - $f(x)=\left \{\begin{matrix} \lambda e^{-\lambda x} & ,x>0 \\ 0 & ,x \leq 0 \end {matrix} \right.$

  - $F(x)= \left \{\begin{matrix} 1 - e^{-\lambda x} & ,x>0 \\ 0 & ,x \leq 0 \end {matrix} \right.$

  - 期望：$E(X) = 1/\lambda$

  - 方差：$D(X) = E(X^2)-[E(X)]^2 = 2/\lambda^2 - 1/\lambda^2$

  - 指数分布是唯一一个具有无记忆性的连续分布

  - > $P(x > t_0 + t | x > t_0) = P(x>t)$

  - > 若旅客进入机场的时间间隔X符合指数分布，已知10分钟内没有旅客进入，求未来2分钟内也没有旅客进入的概率。$P(X>(10+2)|X>10) = P(X>2)$。

- ### 4.4 正态分布（Normal）（高斯分布、误差分布）

  - 记为：$X \sim N(\mu ,\sigma ^2)$

  - $f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-u)^2}{2\sigma ^2}}$

  - 期望：$E(X) = \mu$

  - 方差：$D(X) = \sigma^2$

  - $\mu$为位置参数，$\sigma$为尺度参数，越小越瘦高

  - 根据中心极限定理，多个未知分布的和可用正态分布近似

  - 标准正态：$Z \sim N(0,1)$，$\varphi(z)$，$\Phi(z)$

  - 若 $Y=aX+b$，$X \sim (\mu, \sigma)$，则$Y \sim (a\mu+b, a^2\sigma^2)$

  - 若 $X \sim N(\mu_1 ,\sigma_1 ^2)$，$Y \sim N(\mu_2 ,\sigma_2 ^2)$，并且 $X$ 与 $Y$ 相互独立，则$Z=X+Y \sim N(\mu_1+\mu_2 ,\sigma_1 ^2 + \sigma_2 ^2)$

------

## 5. 第五周

- ## 5.1 二元随机变量分布律

  - 分布律：$P(X=x_i,Y=y_j) = p_{ij}$

  - 边际分布：$P(X=x_i) = p_{i\cdot}$，$P(Y=y_j) = p_{\cdot j}$

  - 条件分布：$P(X=x_i|Y=y_j) = \frac{p_{ij}}{p_{\cdot j}}$

- ## 5.2 二元随机变量分布函数

  - 分布函数：$F(x,y) = P(X \leq x, Y \leq y)$
  > $P(x_1 < X \leq x_2, y_1 < Y \leq y_2) = F(x_2,y_2) - F(x_1,y_2) - F(x_2,y_1) + F(x_1,y_1)$

  - 边际分布函数：$F_X(x) = \lim_{y \to +\infty}F(x,y)$，$F_Y(y) = \lim_{x \to +\infty}F(x,y)$

  - 条件分布函数：$F_{X|Y}(x|y) = P(X \leq x, Y=y)$

- ## 5.3 二元随机变量概率密度

  - 联合概率密度：$F(x,y) = \int_{-\infty}^{x} \int_{-\infty}^{y} f(u,v)dudv$，$\frac{\partial ^2 F(x,y)}{\partial x \partial y}=f(x,y)$

------

## 6. 第六周

- ## 6.1 二元随机变量边际概率密度和条件概率密度

  - 边际概率密度：$f_X(x) = \int_{-\infty}^{+\infty} f(x,y)dy$，$f_Y(y) = \int_{-\infty}^{+\infty} f(x,y)dx$

  - 条件概率密度：$f_{X|Y}(x|y) = \frac{f(x,y)}{f_Y(y)}$

- ## 6.2 二元均匀分布

  - $f(x,y)= \left \{\begin{matrix} 1/A & ,(x,y)\in D \\ 0 & ,(x,y)\notin D \end {matrix} \right.$，A为区域D的面积
  
  - 其边际分布不是均匀分布，其条件分布是均匀分布

- ## 6.3 二元正态分布

  - 记为：$(X,Y) \sim N(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)$

  - $f(x,y)$ 太长省略

  - 其边际分布、条件分布也是正态分布

  - 当 $\rho=0$，$X$ 与 $Y$ 相互独立（充要条件）

- ## 6.4 随机变量的独立性

  - 随机变量独立：$F(x,y) = F_X(x) \cdot F_Y(y)$

  - 离散型：对一切 $i,j$ 都有 $P(X=x_i, Y=y_j) = P(X=x_i) \cdot P(Y=y_j)$

  - 连续型：对可取范围内的点 $(x,y)$，$f(x,y) = f_X(x) \cdot f_Y(y)$

------

## 7. 第七周

- 7.1 连续型变量 $Z=X+Y$ 的分布

  - 已知 $f(x,y)$，$z=x+y$，求 $F(z)$

  - $f_Z(z) = \int_{-\infty}^{+\infty} f(z-y,y)dy$ 或者 $f_Z(z) = \int_{-\infty}^{+\infty} f(x,z-x)dy$$

  - 当 $x,y$ 相互独立时，出现卷积公式，$f_Z(z) = \int_{-\infty}^{+\infty} f_X(z-y)f_Y(y)dy = \int_{-\infty}^{+\infty} f_X(x)f_Y(z-x)dy$

- 7.2 $M = max(X,Y)$ 和 $N = min(X,Y)$ 的分布

  - $F_{max}(z) = P(M \leq z) = P(X \leq z, Y \leq z)$

  - $F_{max}(z) = P(N \leq z) = 1-P(N > z) = 1 - P(X > z, Y > z)$

------





