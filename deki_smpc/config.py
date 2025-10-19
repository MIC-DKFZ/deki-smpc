from openfhe import CCParamsBGVRNS, GenCryptoContext, PKESchemeFeature

from openfhe import *

parameters = CCParamsBGVRNS()
parameters.SetPlaintextModulus(1146881)  # prime, ~1.15M
parameters.SetMultiplicativeDepth(2)
parameters.SetRingDim(16384)  # m = 32768 divides t-1

crypto_context = GenCryptoContext(parameters)
crypto_context.Enable(PKESchemeFeature.PKE)
crypto_context.Enable(PKESchemeFeature.KEYSWITCH)
crypto_context.Enable(PKESchemeFeature.LEVELEDSHE)


MAX_LENGTH = crypto_context.GetRingDimension() // 2
