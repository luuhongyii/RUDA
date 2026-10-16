// Bessel functions adapted from PyTorch ATen/native/Math.h and cuda/Math.cuh.
/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
// Cephes coefficient tables:
// Copyright (c) 2018, Steven Moshier. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

use super::*;
use ruda_kernel::dsl as kernel_dsl;

#[ruda]
fn j0_pp<F:Float>(x:F)->F {
    let mut p=F::cast_from(7.96936729297347051624e-04f64);
    p=p*x+F::cast_from(8.28352392107440799803e-02f64);
    p=p*x+F::cast_from(1.23953371646414299388e+00f64);
    p=p*x+F::cast_from(5.44725003058768775090e+00f64);
    p=p*x+F::cast_from(8.74716500199817011941e+00f64);
    p=p*x+F::cast_from(5.30324038235394892183e+00f64);
    p=p*x+F::cast_from(9.99999999999999997821e-01f64);
    p
}

#[ruda]
fn j0_pq<F:Float>(x:F)->F {
    let mut p=F::cast_from(9.24408810558863637013e-04f64);
    p=p*x+F::cast_from(8.56288474354474431428e-02f64);
    p=p*x+F::cast_from(1.25352743901058953537e+00f64);
    p=p*x+F::cast_from(5.47097740330417105182e+00f64);
    p=p*x+F::cast_from(8.76190883237069594232e+00f64);
    p=p*x+F::cast_from(5.30605288235394617618e+00f64);
    p=p*x+F::cast_from(1.00000000000000000218e+00f64);
    p
}

#[ruda]
fn j0_qp<F:Float>(x:F)->F {
    let mut p=F::cast_from(-1.13663838898469149931e-02f64);
    p=p*x+F::cast_from(-1.28252718670509318512e+00f64);
    p=p*x+F::cast_from(-1.95539544257735972385e+01f64);
    p=p*x+F::cast_from(-9.32060152123768231369e+01f64);
    p=p*x+F::cast_from(-1.77681167980488050595e+02f64);
    p=p*x+F::cast_from(-1.47077505154951170175e+02f64);
    p=p*x+F::cast_from(-5.14105326766599330220e+01f64);
    p=p*x+F::cast_from(-6.05014350600728481186e+00f64);
    p
}

#[ruda]
fn j0_qq<F:Float>(x:F)->F {
    let mut p=F::cast_from(6.43178256118178023184e+01f64);
    p=p*x+F::cast_from(8.56430025976980587198e+02f64);
    p=p*x+F::cast_from(3.88240183605401609683e+03f64);
    p=p*x+F::cast_from(7.24046774195652478189e+03f64);
    p=p*x+F::cast_from(5.93072701187316984827e+03f64);
    p=p*x+F::cast_from(2.06209331660327847417e+03f64);
    p=p*x+F::cast_from(2.42005740240291393179e+02f64);
    p
}

#[ruda]
fn j0_rp<F:Float>(x:F)->F {
    let mut p=F::cast_from(-4.79443220978201773821e+09f64);
    p=p*x+F::cast_from(1.95617491946556577543e+12f64);
    p=p*x+F::cast_from(-2.49248344360967716204e+14f64);
    p=p*x+F::cast_from(9.70862251047306323952e+15f64);
    p
}

#[ruda]
fn j0_rq<F:Float>(x:F)->F {
    let mut p=F::cast_from(4.99563147152651017219e+02f64);
    p=p*x+F::cast_from(1.73785401676374683123e+05f64);
    p=p*x+F::cast_from(4.84409658339962045305e+07f64);
    p=p*x+F::cast_from(1.11855537045356834862e+10f64);
    p=p*x+F::cast_from(2.11277520115489217587e+12f64);
    p=p*x+F::cast_from(3.10518229857422583814e+14f64);
    p=p*x+F::cast_from(3.18121955943204943306e+16f64);
    p=p*x+F::cast_from(1.71086294081043136091e+18f64);
    p
}

#[ruda]
fn j1_pp<F:Float>(x:F)->F {
    let mut p=F::cast_from(7.62125616208173112003e-04f64);
    p=p*x+F::cast_from(7.31397056940917570436e-02f64);
    p=p*x+F::cast_from(1.12719608129684925192e+00f64);
    p=p*x+F::cast_from(5.11207951146807644818e+00f64);
    p=p*x+F::cast_from(8.42404590141772420927e+00f64);
    p=p*x+F::cast_from(5.21451598682361504063e+00f64);
    p=p*x+F::cast_from(1.00000000000000000254e+00f64);
    p
}

#[ruda]
fn j1_pq<F:Float>(x:F)->F {
    let mut p=F::cast_from(5.71323128072548699714e-04f64);
    p=p*x+F::cast_from(6.88455908754495404082e-02f64);
    p=p*x+F::cast_from(1.10514232634061696926e+00f64);
    p=p*x+F::cast_from(5.07386386128601488557e+00f64);
    p=p*x+F::cast_from(8.39985554327604159757e+00f64);
    p=p*x+F::cast_from(5.20982848682361821619e+00f64);
    p=p*x+F::cast_from(9.99999999999999997461e-01f64);
    p
}

#[ruda]
fn j1_qp<F:Float>(x:F)->F {
    let mut p=F::cast_from(5.10862594750176621635e-02f64);
    p=p*x+F::cast_from(4.98213872951233449420e+00f64);
    p=p*x+F::cast_from(7.58238284132545283818e+01f64);
    p=p*x+F::cast_from(3.66779609360150777800e+02f64);
    p=p*x+F::cast_from(7.10856304998926107277e+02f64);
    p=p*x+F::cast_from(5.97489612400613639965e+02f64);
    p=p*x+F::cast_from(2.11688757100572135698e+02f64);
    p=p*x+F::cast_from(2.52070205858023719784e+01f64);
    p
}

#[ruda]
fn j1_qq<F:Float>(x:F)->F {
    let mut p=F::cast_from(7.42373277035675149943e+01f64);
    p=p*x+F::cast_from(1.05644886038262816351e+03f64);
    p=p*x+F::cast_from(4.98641058337653607651e+03f64);
    p=p*x+F::cast_from(9.56231892404756170795e+03f64);
    p=p*x+F::cast_from(7.99704160447350683650e+03f64);
    p=p*x+F::cast_from(2.82619278517639096600e+03f64);
    p=p*x+F::cast_from(3.36093607810698293419e+02f64);
    p
}

#[ruda]
fn j1_rp<F:Float>(x:F)->F {
    let mut p=F::cast_from(-8.99971225705559398224e+08f64);
    p=p*x+F::cast_from(4.52228297998194034323e+11f64);
    p=p*x+F::cast_from(-7.27494245221818276015e+13f64);
    p=p*x+F::cast_from(3.68295732863852883286e+15f64);
    p
}

#[ruda]
fn j1_rq<F:Float>(x:F)->F {
    let mut p=F::cast_from(6.20836478118054335476e+02f64);
    p=p*x+F::cast_from(2.56987256757748830383e+05f64);
    p=p*x+F::cast_from(8.35146791431949253037e+07f64);
    p=p*x+F::cast_from(2.21511595479792499675e+10f64);
    p=p*x+F::cast_from(4.74914122079991414898e+12f64);
    p=p*x+F::cast_from(7.84369607876235854894e+14f64);
    p=p*x+F::cast_from(8.95222336184627338078e+16f64);
    p=p*x+F::cast_from(5.32278620332680085395e+18f64);
    p
}

#[ruda]
fn y0_yp<F:Float>(x:F)->F {
    let mut p=F::cast_from(1.55924367855235737965e+04f64);
    p=p*x+F::cast_from(-1.46639295903971606143e+07f64);
    p=p*x+F::cast_from(5.43526477051876500413e+09f64);
    p=p*x+F::cast_from(-9.82136065717911466409e+11f64);
    p=p*x+F::cast_from(8.75906394395366999549e+13f64);
    p=p*x+F::cast_from(-3.46628303384729719441e+15f64);
    p=p*x+F::cast_from(4.42733268572569800351e+16f64);
    p=p*x+F::cast_from(-1.84950800436986690637e+16f64);
    p
}

#[ruda]
fn y0_yq<F:Float>(x:F)->F {
    let mut p=F::cast_from(1.04128353664259848412e+03f64);
    p=p*x+F::cast_from(6.26107330137134956842e+05f64);
    p=p*x+F::cast_from(2.68919633393814121987e+08f64);
    p=p*x+F::cast_from(8.64002487103935000337e+10f64);
    p=p*x+F::cast_from(2.02979612750105546709e+13f64);
    p=p*x+F::cast_from(3.17157752842975028269e+15f64);
    p=p*x+F::cast_from(2.50596256172653059228e+17f64);
    p
}

#[ruda]
fn y1_yp<F:Float>(x:F)->F {
    let mut p=F::cast_from(1.26320474790178026440e+09f64);
    p=p*x+F::cast_from(-6.47355876379160291031e+11f64);
    p=p*x+F::cast_from(1.14509511541823727583e+14f64);
    p=p*x+F::cast_from(-8.12770255501325109621e+15f64);
    p=p*x+F::cast_from(2.02439475713594898196e+17f64);
    p=p*x+F::cast_from(-7.78877196265950026825e+17f64);
    p
}

#[ruda]
fn y1_yq<F:Float>(x:F)->F {
    let mut p=F::cast_from(5.94301592346128195359e+02f64);
    p=p*x+F::cast_from(2.35564092943068577943e+05f64);
    p=p*x+F::cast_from(7.34811944459721705660e+07f64);
    p=p*x+F::cast_from(1.87601316108706159478e+10f64);
    p=p*x+F::cast_from(3.88231277496238566008e+12f64);
    p=p*x+F::cast_from(6.20557727146953693363e+14f64);
    p=p*x+F::cast_from(6.87141087355300489866e+16f64);
    p=p*x+F::cast_from(3.97270608116560655612e+18f64);
    p
}

#[ruda]
fn i0_a<F:Float>(x:F)->F {
    let mut a=F::cast_from(-4.41534164647933937950e-18f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(3.33079451882223809783e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.43127984654795469359e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.71539128555513303061e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.16853328779934516808e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(7.67618549860493561688e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.85644678311192946090e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.95505266312963983461e-12f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.72682629144155570723e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(9.67580903537323691224e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.18979560163526290666e-10f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.65982372468238665035e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.30002500998624804212e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(6.04699502254191894932e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.67079385394061173391e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.11738753912010371815e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.41673835845875056359e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.64484480707288970893e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.75419501008210370398e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.88502885095841655729e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.76375574538582365885e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.63947561694133579842e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.32430999505057594430e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.05464603945949983183e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.37374148058994688156e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(4.93052842396707084878e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(-9.49010970480476444210e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.71620901522208775349e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.04682672343198398683e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(6.76795274409476084995e-01f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn i0_b<F:Float>(x:F)->F {
    let mut a=F::cast_from(-7.23318048787475395456e-18f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(-4.83050448594418207126e-18f64);
    p=q;q=a;a=x*q-p+F::cast_from(4.46562142029675999901e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.46122286769746109310e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.82762398051658348494e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.42548561967721913462e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.77256013305652638360e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.81168066935262242075e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(-9.55484669882830764870e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.15056934728722208663e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.54008621752140982691e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.85277838274214270114e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(7.18012445138366623367e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.79417853150680611778e-12f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.32158118404477131188e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.14991652796324136454e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.18891471078464383424e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(4.94060238822496958910e-10f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.39623202570838634515e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.26666899049817806459e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.04891858946906374183e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.89137052083475648297e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(6.88975834691682398426e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.36911647825569408990e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(8.04490411014108831608e-01f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn i1_a<F:Float>(x:F)->F {
    let mut a=F::cast_from(2.77791411276104639959e-18f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(-2.11142121435816608115e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.55363195773620046921e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.10559694773538630805e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(7.60068429473540693410e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.04218550472791168711e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.22379336594557470981e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.98397439776494371520e-12f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.17361862988909016308e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(-6.66348972350202774223e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.62559028155211703701e-10f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.88724975172282928790e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(9.38153738649577178388e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.44505912879632808065e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.00329475355213526229e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(-8.56872026469545474066e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.47025130813767847674e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.32731636560394358279e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(4.78156510755005422638e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.61760815825896745588e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(5.12285956168575772895e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.51357245063125314899e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(4.15642294431288815669e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.05640848946261981558e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.47264490306265168283e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.29459812080949914269e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.02643658689847095384e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.76416518357834055153e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.52587186443633654823e-01f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn i1_b<F:Float>(x:F)->F {
    let mut a=F::cast_from(7.51729631084210481353e-18f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(4.41434832307170791151e-18f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.65030536848935832153e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.20952592199342395980e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.96262899764595013876e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.30820231092092828324e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.88035477551078244854e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.81440307243700780478e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.04202769841288027642e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(4.27244001671195135429e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.10154184277266431302e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.08355111109219731823e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(-7.19855177624590851209e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.03562854414708950722e-12f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.41258074366137813316e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.25260358301548823856e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.89749581235054123450e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.58974346219658380687e-10f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.83538038596423702205e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.63146884688951950684e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.51223623787020892529e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.88256480887769039346e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.10588938762623716291e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(-9.76109749136146840777e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(7.78576235018280120474e-01f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn k0_a<F:Float>(x:F)->F {
    let mut a=F::cast_from(1.37446543561352307156e-16f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(4.25981614279661018399e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.03496952576338420167e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.90451637722020886025e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.53479107902614945675e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.28621210311945178607e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.26461541144692592338e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.59799365153615016266e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.44289899924628486886e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.35327393233902768720e-01f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn k0_b<F:Float>(x:F)->F {
    let mut a=F::cast_from(5.30043377268626276149e-18f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(-1.64758043015242134646e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(5.21039150503902756861e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.67823109680541210385e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(5.51205597852431940784e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.84859337734377901440e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(6.34007647740507060557e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.22751332699166985548e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(8.03289077536357521100e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.98009692317273043925e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.14034058820847496303e-12f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.51459788337394416547e-12f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.85594911495471785253e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(-7.95748924447710747776e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.57739728140030116597e-10f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.69753450938905987466e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(8.57403401741422608519e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.66048989768794782956e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.76681363944501510342e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.83175552271911948767e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.39498137188764993662e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.28495495816278026384e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.56988388573005337491e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.14481013119645005427e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.44030308206595545468e+00f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn k1_a<F:Float>(x:F)->F {
    let mut a=F::cast_from(-7.02386347938628759343e-18f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(-2.42744985051936593393e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(-6.66690169419932900609e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.41148839263352776110e-10f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.21338763073472585583e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.43340614156596823496e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.73028895751305206302e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(-6.97572385963986435018e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.22611180822657148235e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.53155960776544875667e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.52530022733894777053e+00f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn k1_b<F:Float>(x:F)->F {
    let mut a=F::cast_from(-5.75674448366501715755e-18f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(1.79405087314755922667e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.68946255844285935196e-17f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.83809354436663880070e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(-6.05704724837331885336e-16f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.03870316562433424052e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(-7.01983709041831346144e-15f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.47715442448130437068e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(-8.97670518232499435011e-14f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.34841966607842919884e-13f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.28917396095102890680e-12f64);
    p=q;q=a;a=x*q-p+F::cast_from(5.13963967348173025100e-12f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.12996783842756842877e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(9.21831518760500529508e-11f64);
    p=q;q=a;a=x*q-p+F::cast_from(-4.19035475934189648750e-10f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.01504975519703286596e-09f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.03457624656780970260e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(5.74108412545004946722e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.50196060308781257119e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.40648494783721712015e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.93619797416608296024e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.95215518471351631108e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.85781685962277938680e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.03923736576817238437e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.72062619048444266945e+00f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn i1_a_f32<F:Float>(x:F)->F {
    let mut a=F::cast_from(9.38153738649577178388e-09f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(-4.44505912879632808065e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.00329475355213526229e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(-8.56872026469545474066e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(3.47025130813767847674e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.32731636560394358279e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(4.78156510755005422638e-05f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.61760815825896745588e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(5.12285956168575772895e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.51357245063125314899e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(4.15642294431288815669e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.05640848946261981558e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.47264490306265168283e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(-5.29459812080949914269e-02f64);
    p=q;q=a;a=x*q-p+F::cast_from(1.02643658689847095384e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.76416518357834055153e-01f64);
    p=q;q=a;a=x*q-p+F::cast_from(2.52587186443633654823e-01f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn i1_b_f32<F:Float>(x:F)->F {
    let mut a=F::cast_from(-3.83538038596423702205e-09f64);let mut p=F::new(0.0);let mut q=F::new(0.0);
    p=q;q=a;a=x*q-p+F::cast_from(-2.63146884688951950684e-08f64);
    p=q;q=a;a=x*q-p+F::cast_from(-2.51223623787020892529e-07f64);
    p=q;q=a;a=x*q-p+F::cast_from(-3.88256480887769039346e-06f64);
    p=q;q=a;a=x*q-p+F::cast_from(-1.10588938762623716291e-04f64);
    p=q;q=a;a=x*q-p+F::cast_from(-9.76109749136146840777e-03f64);
    p=q;q=a;a=x*q-p+F::cast_from(7.78576235018280120474e-01f64);
    F::new(0.5)*(a-p)
}

#[ruda]
fn i0<F:Float>(input:F,#[comptime] scaled:bool)->F {
    let x=input.abs();let mut result=F::new(0.0);
    if x<=F::new(8.0) {
        let p=i0_a::<F>(x/F::new(2.0)-F::new(2.0));
        if comptime!(scaled) {result=p;}else {result=x.exp()*p;}
    } else {
        let p=i0_b::<F>(F::new(32.0)/x-F::new(2.0));
        if comptime!(scaled) {result=p/x.sqrt();}else {result=x.exp()*p/x.sqrt();}
    }
    result
}

#[ruda]
fn i1<F:Float>(input:F,#[comptime] scaled:bool,#[comptime] wide:bool,#[comptime] modified:bool)->F {
    let x=input.abs();let mut result=F::new(0.0);
    if x<=F::new(8.0) {
        let t=x/F::new(2.0)-F::new(2.0);let mut p=F::new(0.0);
        if comptime!(scaled && !wide) {p=i1_a_f32::<F>(t);}else {p=i1_a::<F>(t);}
        if comptime!(scaled) {result=p*x;}
        else if comptime!(modified) {result=p*x*x.exp();}
        else {result=x.exp()*x*p;}
    } else {
        let t=F::new(32.0)/x-F::new(2.0);let mut p=F::new(0.0);
        if comptime!(scaled && !wide) {p=i1_b_f32::<F>(t);}else {p=i1_b::<F>(t);}
        if comptime!(scaled) {result=p/x.sqrt();}else {result=x.exp()*p/x.sqrt();}
    }
    if input<F::new(0.0) {result=-result;}
    result
}

#[ruda]
fn far_jy<F:Float>(x:F,#[comptime] first:bool,#[comptime] second:bool)->F {
    let w=F::new(5.0)/x;let mut t=F::new(25.0)/(x*x);
    let mut pp=F::new(0.0);let mut pq=F::new(0.0);
    let mut qp=F::new(0.0);let mut qq=F::new(0.0);
    let mut phase=x-F::cast_from(0.785398163397448309615660845819875721f64);
    if comptime!(first) {
        t=w*w;phase=x-F::cast_from(2.356194490192344928846982537459627163f64);
        pp=j1_pp::<F>(t);pq=j1_pq::<F>(t);qp=j1_qp::<F>(t);qq=j1_qq::<F>(t);
    } else {
        pp=j0_pp::<F>(t);pq=j0_pq::<F>(t);qp=j0_qp::<F>(t);qq=j0_qq::<F>(t);
    }
    let mut result=F::new(0.0);
    if comptime!(second) {result=pp/pq*phase.sin()+w*(qp/qq)*phase.cos();}
    else {result=pp/pq*phase.cos()-w*(qp/qq)*phase.sin();}
    result*F::cast_from(0.797884560802865355879892119868763737f64)/x.sqrt()
}

#[ruda]
fn j<F:Float>(input:F,#[comptime] first:bool)->F {
    let mut x=input;if x<F::new(0.0) {x=-x;}
    let s=x*x;let mut result=F::new(0.0);
    if x<=F::new(5.0) {
        if comptime!(first) {
            result=j1_rp::<F>(s)/j1_rq::<F>(s)*x*(s-F::cast_from(1.46819706421238932572e+01f64))*(s-F::cast_from(4.92184563216946036703e+01f64));
        } else if x<F::cast_from(0.00001f64) {
            result=F::new(1.0)-s/F::new(4.0);
        } else {
            result=(s-F::cast_from(5.78318596294678452118e+00f64))*(s-F::cast_from(3.04712623436620863991e+01f64))*j0_rp::<F>(s)/j0_rq::<F>(s);
        }
    } else {result=far_jy::<F>(x,first,false);}
    if comptime!(first) {if input<F::new(0.0) {result=-result;}}
    result
}

#[ruda]
fn y<F:Float>(x:F,#[comptime] first:bool)->F {
    let mut result=F::new(f32::NAN);
    if x==F::new(0.0) {result=F::new(f32::NEG_INFINITY);}
    else if x>F::new(0.0) {
        if x<=F::new(5.0) {
            let s=x*x;let factor=F::cast_from(0.636619772367581343075535053490057448f64);
            if comptime!(first) {
                result=x*(y1_yp::<F>(s)/y1_yq::<F>(s))+factor*(j::<F>(x,true)*x.ln()-F::new(1.0)/x);
            } else {result=y0_yp::<F>(s)/y0_yq::<F>(s)+factor*x.ln()*j::<F>(x,false);}
        } else {result=far_jy::<F>(x,first,true);}
    }
    result
}

#[ruda]
fn k<F:Float>(x:F,#[comptime] first:bool,#[comptime] scaled:bool)->F {
    let mut result=F::new(f32::NAN);
    if x==F::new(0.0) {result=F::new(f32::INFINITY);}
    else if x>F::new(0.0) {
        if x<=F::new(2.0) {
            let t=x*x-F::new(2.0);
            if comptime!(first) {
                result=(F::new(0.5)*x).ln()*i1::<F>(x,false,true,true)+k1_a::<F>(t)/x;
            } else if comptime!(scaled) {
                result=k0_a::<F>(t)-(F::new(0.5)*x).ln()*i0::<F>(x,false);
            } else {
                result=F::cast_from(f64::cast_from(k0_a::<F>(t))-(0.5f64*f64::cast_from(x)).ln()*f64::cast_from(i0::<F>(x,false)));
            }
            if comptime!(scaled) {result*=x.exp();}
        } else {
            let t=F::new(8.0)/x-F::new(2.0);let mut p=F::new(0.0);
            if comptime!(first) {p=k1_b::<F>(t);}else {p=k0_b::<F>(t);}
            if comptime!(scaled) {result=p/x.sqrt();}else {result=(-x).exp()*p/x.sqrt();}
        }
    }
    result
}

#[ruda]
fn spherical_j0<F:Float>(x:F)->F {
    let mut result=F::new(0.0);
    if x.abs()==F::new(f32::INFINITY) {result=F::new(0.0);}
    else if x.abs()<F::new(0.5) {
        let s=x*x;let one=F::new(1.0);
        result=one+s*(-one/F::new(6.0)+s*(one/F::new(120.0)+s*(-one/F::new(5040.0)+s*(one/F::new(362880.0)+s*(-one/F::cast_from(39916800.0f64)+s*(one/F::cast_from(6227020800.0f64)))))));
    } else {result=x.sin()/x;}
    result
}

#[ruda]
pub(super) fn evaluate<F:Float>(x:F,#[comptime] op:u32,#[comptime] wide:bool)->F {
    let mut result=F::new(f32::NAN);
    if comptime!(op==233 || op==241) {result=i0::<F>(x,false);}
    else if comptime!(op==234) {result=i0::<F>(x,true);}
    else if comptime!(op==235) {result=i1::<F>(x,false,wide,false);}
    else if comptime!(op==236) {result=i1::<F>(x,true,wide,false);}
    else if comptime!(op==237 || op==238) {result=j::<F>(x,op==238);}
    else if comptime!(op==239 || op==240) {result=y::<F>(x,op==240);}
    else if comptime!(op==242) {result=i1::<F>(x,false,wide,true);}
    else if comptime!(op>=243 && op<=246) {result=k::<F>(x,op==244 || op==246,op>=245);}
    else if comptime!(op==247) {result=spherical_j0::<F>(x);}
    result
}

