use std::fs::File;
use std::io::{self, Write};
use std::ops::Range;
use std::f64;

use rand::random;

use vec3::{vec3, Vec3};

mod vec3;

fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        return min;
    }

    if x > max {
        return max;
    }

    return x;
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
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(w, "P3")?; // means that colors are in ASCII
        writeln!(w, "{} {}", self.width, self.height)?;
        writeln!(w, "255")?; // max color value

        for line in self.data.chunks(self.width) {
            for pixel in line {
                let r = 256.0 * clamp(pixel.x, 0.0, 0.999);
                let g = 256.0 * clamp(pixel.y, 0.0, 0.999);
                let b = 256.0 * clamp(pixel.z, 0.0, 0.999);
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
    let t = 0.5 * (unit.y + 1.0); // scale [-1; 1] -> [0; 1]

    // blend white + blue from bottom to top
    Vec3::white() * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
}

struct Sphere {
    center: Vec3,
    radius: f64,
    material: Material,
}

// Information about ray hit
#[derive(Debug, Clone)]
struct HitRecord {
    t: f64,                 // "time" of hit
    p: Vec3,                // point at which hit have happened
    normal: Vec3,           // normal to the surface at point P
    front_face: bool,       // true if ray is outside the object
    material: Material,     // material of hit target
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
        let half_b = oc.dot(ray.direction());
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
            let front_face = ray.direction().dot(outward_normal).is_sign_negative();

            Some(HitRecord {
                t,
                p,
                normal: if front_face {
                    outward_normal
                } else {
                    -outward_normal
                },
                front_face,
                material: self.material.clone(),
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

impl<T> Hit for Vec<T>
where
    T: Hit,
{
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

// trait Material {
//     fn scatter(&self, ray: Ray, hit: HitRecord) -> Option<(Vec3, Ray)>; // (attenuation, scattered)
// }

#[derive(Debug, Clone)]
struct Material {
    kind: MaterialKind,
    albedo: Vec3,
    fuzz: f64,
}

#[derive(Debug, Clone, Copy)]
enum MaterialKind {
    Lambertian,
    Metal,
}

impl Material {
    fn scatter(&self, ray: Ray, at: Vec3, normal: Vec3) -> Option<(Vec3, Ray)> {
        match self.kind {
            MaterialKind::Lambertian => {
                let direction = normal + random_unit_vec3();
                let scattered = Ray {
                    origin: at,
                    direction
                };
                Some((self.albedo, scattered))
            },
            MaterialKind::Metal => {
                let reflected = reflect(ray.direction.unit(), normal);
                let scattered = Ray {
                    origin: at,
                    direction: reflected + self.fuzz * random_in_unit_sphere(),
                };

                if scattered.direction().dot(normal) > 0.0 {
                    Some((self.albedo, scattered))
                } else {
                    None
                }
            }
        }
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
            direction: self.lower_left_corner + u * self.horizontal + v * self.vertical
                - self.origin,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            lower_left_corner: vec3(-2.0, -1.0, -1.0),
            horizontal: vec3(4.0, 0.0, 0.0),
            vertical: vec3(0.0, 2.0, 0.0),
            origin: Vec3::default(),
        }
    }
}

fn reflect(v: Vec3, normal: Vec3) -> Vec3 {
    v - 2.0 * v.dot(normal) * normal
}

// cos(x) distribution
fn random_unit_vec3() -> Vec3 {
    let a = random::<f64>() * 2.0 * f64::consts::PI;
    let z = -1.0 + random::<f64>() * 2.0;
    let r = (1.0 - z).sqrt();

    vec3(r * f64::cos(a), r * f64::sin(a), z)
}

fn random_vec3(min: f64, max: f64) -> Vec3 {
    Vec3::splat(min) + vec3(random::<f64>(), random::<f64>(), random::<f64>()) * (max - min)
}

fn random_in_unit_sphere() -> Vec3 {
    loop {
        let v = random_vec3(-1.0, 1.0);
        if v.length_squared() < 1.0 {
            break v;
        }
    }
}

// fn random_in_hemisphere(normal: Vec3) -> Vec3 {
//     let in_unit = random_in_unit_sphere();
//     if in_unit.dot(normal) >= 0.0 { // In the same hemisphere as normal
//         in_unit
//     } else {
//         -in_unit
//     }
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut image = Image::new(200, 100);

    let samples_per_pixel = 100;
    let camera = Camera::default();
    let world = vec![
        Sphere {
            center: vec3(0.0, 0.0, -1.0),
            radius: 0.5,
            material: Material {
                kind: MaterialKind::Lambertian,
                albedo: vec3(0.7, 0.3, 0.3),
                fuzz: 1.0,
            },
        },
        Sphere {
            center: vec3(0.0, -100.5, -1.0),
            radius: 100.0,
            material: Material {
                kind: MaterialKind::Lambertian,
                albedo: vec3(0.8, 0.8, 0.0),
                fuzz: 1.0,
            }
        },
        Sphere {
            center: vec3(1.0, 0.0, -1.0),
            radius: 0.5,
            material: Material {
                kind: MaterialKind::Metal,
                albedo: vec3(0.8, 0.6, 0.2),
                fuzz: 1.0,
            }
        },
        Sphere {
            center: vec3(-1.0, 0.0, -1.0),
            radius: 0.5,
            material: Material {
                kind: MaterialKind::Metal,
                albedo: vec3(0.8, 0.8, 0.8),
                fuzz: 0.3,
            }
        }
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
                let mut ray = camera.ray(u, v);

                const MAX_BOUNCES: usize = 50;

                let mut i = 0;
                let mut attenuation = vec3(1.0, 1.0, 1.0);
                let color = loop {
                    if i > MAX_BOUNCES {
                        break Vec3::black();
                    }

                    match world.hit(ray, 0.001..std::f64::INFINITY) {
                        Some(hit) => {
                            match hit.material.scatter(ray, hit.p, hit.normal) {
                                Some((attenuation_increment, scattered)) => {
                                    attenuation *= attenuation_increment;
                                    ray = scattered;
                                    i += 1;
                                },
                                None => break Vec3::black(),
                            }

                        },
                        None => break background_color(ray),
                    }
                };

                pixel += color * attenuation;
            }


            let color = pixel / samples_per_pixel as f64;

            image.set(
                (i, image.height() - j - 1),
                color.sqrt(), // gamma correction
            );
        }
    }

    let mut out = File::create("out.ppm")?;
    image.write(&mut out)?;
    Ok(())
}
