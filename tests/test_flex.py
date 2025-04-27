from models.llama import Model, ModelArgs

# Create model args with flex attention
args = ModelArgs(
    model_type="llama",
    hidden_size=128,
    num_hidden_layers=2,
    intermediate_size=512,
    num_attention_heads=4,
    use_flash_attention=False,
    use_flex_attention=True,
    flash_block_size=128
)

# Try to create the model
model = Model(args)
print("Model created successfully with FlexAttention")
print(f"Parameters: use_flash_attention={args.use_flash_attention}, use_flex_attention={args.use_flex_attention}")
