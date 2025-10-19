from openfhe import CCParamsBFVRNS, GenCryptoContext, PKESchemeFeature

parameters = CCParamsBFVRNS()
parameters.SetPlaintextModulus(1032193)
parameters.SetMultiplicativeDepth(2)

cc = GenCryptoContext(parameters)

cc.Enable(PKESchemeFeature.PKE)
cc.Enable(PKESchemeFeature.KEYSWITCH)
cc.Enable(PKESchemeFeature.LEVELEDSHE)

MAX_LENGTH = cc.GetRingDimension() // 2
