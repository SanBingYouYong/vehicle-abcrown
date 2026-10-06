-- Global robustness of the MNIST classifier: for *every* valid image, not
-- just dataset images, an epsilon perturbation changes every output score by
-- at most delta. Both network applications have free inputs.
--
-- This is much harder than robustness around dataset images: α,β-CROWN's
-- bounds over the whole input space are very loose for this network.

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

@property
globalRobust : Bool
globalRobust = forall image perturbation .
  let perturbedImage = image - perturbation in
  validImage image and boundedByEpsilon perturbation and validImage perturbedImage =>
    (forall i . let d = classifier perturbedImage ! i - classifier image ! i in
       -delta <= d and d <= delta)
