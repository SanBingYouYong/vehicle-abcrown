-- Standard robustness of the MNIST classifier around dataset images:
-- perturbing an image by at most epsilon in every pixel changes every output
-- score by at most delta, i.e. |f(x') - f(x)| <= delta.
--
-- `classifier image` has a fixed input (it comes from the dataset), so the
-- backend evaluates it once and only `classifier perturbedImage` is verified.

type Image = Tensor Real [28, 28]

validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

@network
classifier : Image -> Tensor Real [10]

@parameter
epsilon : Real

@parameter
delta : Real

boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

standardRobustAround : Image -> Bool
standardRobustAround image = forall perturbation .
  let perturbedImage = image - perturbation in
  boundedByEpsilon perturbation and validImage perturbedImage =>
    (forall i . let d = classifier perturbedImage ! i - classifier image ! i in
       -delta <= d and d <= delta)

@parameter(infer=True)
n : Nat

@dataset
images : Vector Image n

@property
standardRobust : Vector Bool n
standardRobust = foreach k . standardRobustAround (images ! k)
