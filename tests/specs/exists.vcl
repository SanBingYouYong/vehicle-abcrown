-- Existential properties: Vehicle reports a solver "sat" as a witness.

type Input = Tensor Real [2]

@network
toy : Input -> Tensor Real [2]

@parameter
c : Real

validInput : Input -> Bool
validInput x = forall i . 0 <= x ! i <= 1

@property
reachable : Bool
reachable = exists x . validInput x and toy x ! 0 >= c

@property
closeAndDifferent : Bool
closeAndDifferent = exists x y . validInput x and validInput y and
  (forall i . -0.05 <= x ! i - y ! i <= 0.05) and toy x ! 0 - toy y ! 0 >= c
