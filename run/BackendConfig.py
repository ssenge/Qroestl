# IonQ_token = 'put-your-token-here'
# IBMQ: see utils.SaveIBMQToken.py
# DWave: use the Dwave CLI tool to set token
# Braket: use the AWS CLI tool to set token

BRAKET_S3_BUCKET = ('amazon-braket-765de053a863', 'test')

qdev = (
    # QuantumInstance(BasicAer.get_backend('statevector_simulator'))
    # IBMQ.load_account().get_backend('ibmq_manila')  # make sure you have an IBMQ account and the APItoken is locally saved
    # IonQProvider(token=IonQ_token).get_backend("ionq_simulator")  # make sure you have an IonQ API token
    # IonQProvider(token=IonQ_token).get_backend("ionq_qpu")  # make sure you have an IonQ API token
    'arn:aws:braket:::device/qpu/d-wave/Advantage_system4'
)