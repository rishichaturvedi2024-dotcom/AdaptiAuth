"""
PAD Class Mapping & Logit Diagnostic
=====================================
Tests the trained CNN on 3 inputs with ONE preprocessing function.
Shows raw pre-sigmoid logits, sigmoid probability, and class mapping.
"""
import os, sys, torch, cv2, numpy as np
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml.liveness.pad import SimplePADCNN

def preprocess_image(path_or_array):
    """SINGLE preprocessing function used for ALL three test cases."""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # PIL -> [0,1] float tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if isinstance(path_or_array, str):
        img = Image.open(path_or_array).convert('RGB')
    else:
        # numpy BGR array (from cv2)
        rgb = path_or_array[..., ::-1].copy()
        img = Image.fromarray(rgb)
    return transform(img).unsqueeze(0)  # (1, 3, 128, 128)

def diagnose(label, tensor, model, device):
    """Run model and print raw logits, sigmoid, class mapping."""
    tensor = tensor.to(device)

    # Hook into the pre-sigmoid linear layer to get raw logits
    # The classifier is: Linear(64,32) -> ReLU -> Linear(32,1) -> Sigmoid
    # We need the value BEFORE Sigmoid
    pre_sigmoid_val = None
    def hook_fn(module, input, output):
        nonlocal pre_sigmoid_val
        pre_sigmoid_val = input[0].item()  # Sigmoid input = Linear output

    # Register hook on the Sigmoid layer (classifier[3])
    hook = model.classifier[3].register_forward_hook(hook_fn)

    with torch.no_grad():
        output = model(tensor)
        sigmoid_prob = output.item()

    hook.remove()

    # Class mapping from train_pad.py:
    #   ClientRaw (live)     -> label 1.0
    #   ImposterRaw (spoof)  -> label 0.0
    # Model output: Sigmoid -> P(live)
    # So: sigmoid > 0.5 -> LIVE, sigmoid < 0.5 -> SPOOF

    class_idx = 1 if sigmoid_prob > 0.5 else 0
    class_name = "LIVE" if class_idx == 1 else "SPOOF"
    live_probability = sigmoid_prob
    displayed_pad_score = sigmoid_prob  # This is what live_demo.py shows

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Tensor shape:          {tuple(tensor.shape)}")
    print(f"  Tensor mean:           {tensor.mean().item():.6f}")
    print(f"  Tensor std:            {tensor.std().item():.6f}")
    print(f"  Tensor min:            {tensor.min().item():.6f}")
    print(f"  Tensor max:            {tensor.max().item():.6f}")
    print(f"  ---")
    print(f"  Raw logit (pre-sig):   {pre_sigmoid_val:.6f}")
    print(f"  Sigmoid probability:   {sigmoid_prob:.6f}")
    print(f"  Class index:           {class_idx}")
    print(f"  Class name:            {class_name}")
    print(f"  P(live):               {live_probability:.6f}")
    print(f"  P(spoof):              {1 - live_probability:.6f}")
    print(f"  Displayed PAD score:   {displayed_pad_score:.2f}")
    print(f"  Dashboard would show:  {'Live' if displayed_pad_score > 0.5 else 'Spoof'} ({displayed_pad_score:.2f})")

    return sigmoid_prob

def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    model_path = os.path.join(base, 'data', 'models', 'pad_cnn.pth')

    nuaa_live = os.path.join(base, 'data', 'nuaa', 'raw', 'ClientRaw', '0001', '0001_00_00_01_0.jpg')
    nuaa_spoof = os.path.join(base, 'data', 'nuaa', 'raw', 'ImposterRaw', '0001', '0001_00_00_01_0.jpg')
    webcam = os.path.join(base, 'debug_webcam_frame.jpg')

    device = torch.device('cpu')
    model = SimplePADCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Loaded PAD CNN weights.")

    print("\nClass mapping from train_pad.py:")
    print("  ClientRaw (live)    -> label 1.0")
    print("  ImposterRaw (spoof) -> label 0.0")
    print("  Sigmoid output = P(live)")
    print("  > 0.5 = LIVE, < 0.5 = SPOOF")

    # ---- All three use the SAME preprocess_image() function ----
    print("\n\nALL THREE TESTS USE IDENTICAL PREPROCESSING:")
    print("  PIL.open().convert('RGB') -> Resize(128) -> ToTensor -> Normalize(ImageNet)")

    t1 = preprocess_image(nuaa_live)
    s1 = diagnose("1. NUAA BONA FIDE (ClientRaw) -- should be LIVE", t1, model, device)

    t2 = preprocess_image(nuaa_spoof)
    s2 = diagnose("2. NUAA SPOOF (ImposterRaw) -- should be SPOOF", t2, model, device)

    if os.path.exists(webcam):
        t3 = preprocess_image(webcam)
        s3 = diagnose("3. REAL WEBCAM FRAME (full frame, not MTCNN crop)", t3, model, device)
    else:
        print(f"\n  WEBCAM FILE NOT FOUND: {webcam}")
        s3 = None

    # ---- Now test with MTCNN face crop (the OLD broken path) ----
    print("\n\n" + "#" * 60)
    print("BONUS: OLD LIVE PATH (MTCNN face crop -- the broken pipeline)")
    print("#" * 60)
    from ml.facial.alignment import FaceAligner
    aligner = FaceAligner()

    for label, path in [("NUAA LIVE (MTCNN crop)", nuaa_live),
                         ("WEBCAM (MTCNN crop)", webcam)]:
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        face = aligner.align(img)
        if face is None:
            print(f"\n  {label}: MTCNN found no face")
            continue
        # This is what the OLD live_demo.py did:
        t = face.clone()
        if t.dim() == 3: t = t.unsqueeze(0)
        if t.min() < 0: t = (t + 1.0) / 2.0
        crop_tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t = crop_tf(t)
        diagnose(f"{label} -- OLD broken path", t, model, device)

    # ---- VERDICT ----
    print("\n\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if s1 is not None and s1 > 0.5 and s2 is not None and s2 < 0.5:
        print("  Model correctly classifies NUAA live/spoof with training preprocessing.")
        print("  Class mapping is CORRECT (no inversion).")
        if s3 is not None and s3 > 0.5:
            print(f"  Webcam full frame classified as LIVE ({s3:.4f}).")
            print("\n  ANSWER: (B) Preprocessing mismatch.")
            print("  The OLD live path sent MTCNN face crops (160x160, face-only)")
            print("  into a CNN trained on full uncropped NUAA images.")
            print("  The FIX (detect_frame) sends the full frame and works correctly.")
            print("  NO RETRAINING NEEDED.")
        else:
            print("  ANSWER: (C) Webcam domain shift -- would need investigation.")
    else:
        print("  Model does NOT correctly classify NUAA -- check training.")

if __name__ == "__main__":
    main()
