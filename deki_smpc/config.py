from openfhe import *

parameters = CCParamsCKKSRNS()
parameters.SetMultiplicativeDepth(5)
parameters.SetScalingModSize(50)
parameters.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)

cc = GenCryptoContext(parameters)

print(f"CKKS scheme is using ring dimension {cc.GetRingDimension()}\n")
print(f"CKKS scheme is using batch size {cc.GetBatchSize()}\n")

cc.Enable(PKESchemeFeature.PKE)
cc.Enable(PKESchemeFeature.KEYSWITCH)
cc.Enable(PKESchemeFeature.LEVELEDSHE)

MAX_LENGTH = cc.GetRingDimension() // 2
