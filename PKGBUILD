# Maintainer: Will Handley <wh260@cam.ac.uk>
pkgname=python-jaxwavelets
pkgver=0.1.13
pkgrel=1
pkgdesc="JAX-native wavelet transforms"
arch=('any')
url="https://github.com/handley-lab/jaxwavelets"
license=('MIT')
depends=('python' 'python-jax')
makedepends=('python-build' 'python-installer' 'python-setuptools')
checkdepends=('python-pytest' 'python-pywavelets')
source=("$pkgname-$pkgver.tar.gz::$url/archive/v$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
    cd jaxwavelets-$pkgver
    python -m build --wheel --no-isolation
}

check() {
    cd jaxwavelets-$pkgver
    JAX_ENABLE_X64=1 python -m pytest jaxwavelets/tests/ -x --tb=short
}

package() {
    cd jaxwavelets-$pkgver
    python -m installer --destdir="$pkgdir" dist/*.whl
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
