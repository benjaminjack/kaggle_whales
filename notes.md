# Notes
- Drop out (0.5) w/ 512 units in dense layer did not work at all. Has no regularizing effect and ruins performance.

## Things to try
- [x] Augmentation (~75-80% acc, no overfitting)
- [ ] Learning rate decay
- [ ] Image aspect ration of 2:1 or 3:2 that better represents landscape input data
- [ ] Fine-tuning with augmentation
- [ ] Triplet loss
- [ ] Resnet or InceptionV3 base
- [ ] Pull out output from dense (2048) layer and cluster
