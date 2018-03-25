# Notes
- Drop out (0.5) w/ 512 units in dense layer did not work at all. Has no regularizing effect and ruins performance.
- Resnet50 base gives a val_acc of 0.5 that never moves. Could be because of batch normalization? Inception V3 does more or less the same thing. Will have to test different conv bases with the cats and dogs set to see what's going on.

## Things to try
- [x] Augmentation (~75-80% acc, no overfitting)
- [ ] Learning rate decay
- [ ] Image aspect ration of 2:1 or 3:2 that better represents landscape input data
- [ ] Fine-tuning with augmentation
- [ ] Triplet loss
- [ ] Resnet or InceptionV3 base
- [ ] Pull out output from dense (2048) layer and cluster
