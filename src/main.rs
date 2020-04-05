use std::fs::File;
use std::io::{self, Write};
use std::ops::{Add, AddAssign, Sub, SubAssign, Div, DivAssign, Mul, MulAssign, Neg, Range};

use rand::random;

fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        return min;
    }

    if x > max {
        return max;
    }

    return x;
}

#[derive(Debug, Clone, Copy)]
pub struct Vec3([f64; 3]);

impl Vec3 {
    pub fn white() -> Self {
        Vec3([1.0, 1.0, 1.0])
    }

    pub fn red() -> Self {
        Vec3([1.0, 0.0, 0.0])
    }

    pub fn unit(&self) -> Self {
        *self / self.length()
    }

    pub fn x(&self) -> f64 {
        self.0[0]
    }

    pub fn y(&self) -> f64 {
        self.0[1]
    }

    pub fn z(&self) -> f64 {
        self.0[2]
    }

    pub fn length_squared(&self) -> f64 {
        self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]
    }

    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3([0.0; 3])
    }
}

impl From<(f64, f64, f64)> for Vec3 {
    fn from((x, y, z): (f64, f64, f64)) -> Vec3 {
        Vec3([x, y, z])
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self {
        Vec3([-self.0[0], -self.0[1], -self.0[1]])
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3([self.x() + rhs.x(), self.y() + rhs.y(), self.z() + rhs.z()])
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3([self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z()])
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Vec3 {
        Vec3([self.x() * rhs, self.y() * rhs, self.z() * rhs])
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Vec3 {
        Vec3([self.x() / rhs, self.y() / rhs, self.z() / rhs])
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, rhs: f64) {
        self.0[0] /= rhs;
        self.0[1] /= rhs;
        self.0[2] /= rhs;
    }
}

pub fn dot(lhs: Vec3, rhs: Vec3) -> f64 {
    lhs.x() * rhs.x() + lhs.y() * rhs.y() + lhs.z() * rhs.z()
}

pub fn cross(lhs: Vec3, rhs: Vec3) -> Vec3 {
    Vec3([
        lhs.y() * rhs.z() - lhs.z() * rhs.y(),
        lhs.z() * rhs.x() - lhs.x() * rhs.z(),
        lhs.x() * rhs.y() - rhs.y() * rhs.x(),
    ])
}

pub struct Image {
    data: Vec<Vec3>,
    width: usize,
    height: usize,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Image {
            data: vec![Vec3::default(); width * height],
            width,
            height,
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn get(&self, (x, y): (usize, usize)) -> Vec3 {
        let index = y * self.width + x;
        self.data[index]
    }

    pub fn set(&mut self, (x, y): (usize, usize), pixel: Vec3) {
        let index = y * self.width + x;
        self.data[index] = pixel;
    }

    // write as PPM
    pub fn write<W: Write>(&self, w: &mut W, samples_per_pixel: usize) -> io::Result<()> {
        writeln!(w, "P3")?; // means that colors are in ASCII
        writeln!(w, "{} {}", self.width, self.height)?;
        writeln!(w, "255")?; // max color value

        let scale = 1.0 / samples_per_pixel as f64;
        for line in self.data.chunks(self.width) {
            for pixel in line {
                let r = 256.0 * clamp(pixel.x() * scale, 0.0, 0.999);
                let g = 256.0 * clamp(pixel.y() * scale, 0.0, 0.999);
                let b = 256.0 * clamp(pixel.z() * scale, 0.0, 0.999);
                write!(w, "{} {} {} ", r as u8, g as u8, b as u8)?;
            }
            writeln!(w, "")?;
        }

        Ok(())
    }
}

// Ray which starts at point |origin| and moving to direction |direction|
#[derive(Debug, Clone, Copy)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn origin(&self) -> Vec3 {
        self.origin
    }

    fn direction(&self) -> Vec3 {
        self.direction
    }

    // Calculate position of ray at "time" t
    fn at(&self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}

// determines the "background" color
fn background_color(ray: Ray) -> Vec3 {
    let unit = ray.direction().unit();
    let t = 0.5 * (unit.y() + 1.0); // scale [-1; 1] -> [0; 1]

    // blend white + blue from bottom to top
    Vec3::white() * (1.0 - t) + Vec3([0.5, 0.7, 1.0]) * t
}

struct Sphere {
    center: Vec3,
    radius: f64
}

#[derive(Debug, Clone)]
struct HitRecord {
    t: f64,           // "time" of hit
    p: Vec3,          // point at which hit have happened
    normal: Vec3,     // normal to the surface at point P
    front_face: bool, // true if ray is outside the object
}

trait Hit {
    fn hit(&self, ray: Ray, range: Range<f64>) -> Option<HitRecord>;
}

impl Hit for Sphere {
    // let C - center of sphere of radius R
    // then point P is inside shpere if
    // (p - c) * (p - c)<= R*R  # where * is a dot product
    // in our case, point P could be any on the Ray, so
    // Ray hits the sphere if the equation:
    // (o + d*t - c) * (o + d*t - c) - R*R = 0 is solvable # d = direction, o = origin
    //    =>
    // d*t*d*t + 2*d*t*(o - c) + (o - c) * (o - c) - R*R = 0
    fn hit(&self, ray: Ray, range: Range<f64>) -> Option<HitRecord> {
        let oc = ray.origin() - self.center; // o - c

        // ax^2 + 2bx + c = 0
        let a = ray.direction().length_squared();
        let half_b = dot(oc, ray.direction());
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant.is_sign_negative() {
            return None;
        }

        let root = discriminant.sqrt();
        let t = (-half_b - root) / a;

        let hit = |t| {
            let p = ray.at(t);
            let outward_normal = (p - self.center) / self.radius;
            let front_face = dot(ray.direction(), outward_normal).is_sign_negative();

            Some(HitRecord {
                t,
                p,
                normal: if front_face { outward_normal } else { -outward_normal },
                front_face
            })
        };

        if range.contains(&t) {
            return hit(t);
        }

        let t = (-half_b + root) / a;
        if range.contains(&t) {
            return hit(t);
        }

        None
    }
}

impl<T> Hit for Vec<T> where T: Hit {
    fn hit(&self, ray: Ray, range: Range<f64>) -> Option<HitRecord> {
        let mut record = None;
        let mut max_t = range.end;

        for object in self {
            if let Some(r) = object.hit(ray, range.start..max_t) {
                max_t = r.t;
                record = Some(r.clone());
            }
        }

        record
    }
}

struct Camera {
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    origin: Vec3,
}

impl Camera {
    fn ray(&self, u: f64, v: f64) -> Ray {
        Ray {
            origin: self.origin,
            direction: self.lower_left_corner + u*self.horizontal + v*self.vertical - self.origin,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            lower_left_corner: Vec3([-2.0, -1.0, -1.0]),
            horizontal:  Vec3([4.0, 0.0, 0.0]),
            vertical: Vec3([0.0, 2.0, 0.0]),
            origin: Vec3::default(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut image = Image::new(200, 100);

    let samples_per_pixel = 100;
    let camera = Camera::default();
    let world = vec![
        Sphere {
            center: Vec3([0.0, 0.0, -1.0]),
            radius: 0.5
        },
        Sphere {
            center: Vec3([0.0, -100.5, -1.0]),
            radius: 100.0
        },
    ];

    for j in 0..image.height() {
        // report progress
        println!(
            "Scanlines remaining: {:4} ({:.2}% done)",
            image.height() - j,
            j as f64 / image.height() as f64 * 100.0
        );

        for i in 0..image.width() {
            let mut pixel = Vec3::white();
            for _ in 0..samples_per_pixel {
                let u = (i as f64 + random::<f64>()) / image.width() as f64;
                let v = (j as f64 + random::<f64>()) / image.height() as f64;
                let ray = camera.ray(u, v);

                pixel += match world.hit(ray, 0.0..std::f64::INFINITY) {
                    Some(record) => 0.5 * (record.normal + Vec3([1.0, 1.0, 1.0])),
                    None => background_color(ray)
                };
            }

            image.set((i, image.height() - j - 1), pixel);
        }
    }

    let mut out = File::create("out.ppm")?;
    image.write(&mut out, samples_per_pixel)?;
    Ok(())
}
