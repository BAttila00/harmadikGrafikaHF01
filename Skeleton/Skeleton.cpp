//=============================================================================================
//3. HF
//=============================================================================================
#include "framework.h"

//---------------------------
//Aut?matikus deriv?l?shoz fogjuk haszn?lni
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
//Ezeket is aut?matikus deriv?l?shoz fogjuk haszn?lni
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

//du?lis sz?m template strukt?ra vec2-vel val? parametriz?l?s?t elnevezz?k Dnum2-nek
typedef Dnum<vec2> Dnum2;

//h?ny sor illetve oszlop lesz a fel?letekhez tertoz? teszell?land? egys?gn?gyzetben l?sd 9. diasor 9. dia
const int tessellationLevel = 20;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	// extrinsic
	vec3 wEye;			//kamera pos?ci?
	vec3 wLookat;		//n?zeti ir?ny
	vec3 wVup;			//prefer?lt f?gg?leges ir?ny
	// intrinsic
	float fov;			//f?gg?leges l?t?sz?g
	float asp;			//aspektus ar?ny (a f?gg?leges/a v?zszintes l?t?sz?g ar?nya) (l?sd 9. diasor 17. ?s 26. dia)
	float fp;			//els? v?g?s?k t?vols?ga (l?sd 9. diasor 17. ?s 26. dia)
	float bp;			//h?ts? v?g?s?k t?vols?ga (l?sd 9. diasor 17. ?s 26. dia)
	float fovAngle;
public:
	Camera() {
		fovAngle = 60.0f;
		asp = (float)windowWidth / windowHeight;
		fov = fovAngle * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	}
	/// <summary>
	/// Az?rt transzform?ltunk, hogy a l?that?s?gi feladatot ?s a vet?t?st k?perny? koordin?tarendszerben oldhassuk meg
	/// </summary>
	/// <returns></returns>
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	/// <summary>
	/// Az?rt transzform?ltunk, hogy a l?that?s?gi feladatot ?s a vet?t?st k?perny? koordin?tarendszerben oldhassuk meg
	/// </summary>
	/// <returns></returns>
	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

//---------------------------
struct Material {	//inkrement?lis k?pszint?zisn?l diff?z fel?letekkel dolgozunk t?bbnyire
	//---------------------------

	vec3 kd;	//diff?z visszaver?d?si t?nyez?
	vec3 ks;	//spekul?ris visszaver?d?si t?nyez?
	vec3 ka;	//ambiens visszaver?d?si t?nyez?
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La;	//ambiens f?nyforr?s (itt most minden f?nyforr?shoz felvett k?l?n ambiens f?nyt, lehetne ?gy is h csak egy ambiens f?ny van glob?lisan)
	vec3 Le;	//az adott f?nyforr?s sug?rs?r?s?ge, intenzit?sa, f?nyer?ss?ge stb stb
	vec4 wLightPos; //f?nyforr?s helye		// homogeneous coordinates, can be at ideal point	
	//mivel ez egy homog?n koordin?ta, ?gy lehet az ide?lis pontban (v?gtelenben) (ha az uts? koordin?ta nulla)
	//mivel ha egy pontszer? f?nyforr?s a v?gtelenben van akkor azt ir?nyf?nyforr?snak vessz?k ez?rt ezzel a strukt?r?val a pont ?s ir?ny f?nyforr?sok is kezelhet?ek
};

//---------------------------
//egy sakkt?blatext?r?t val?s?t meg
class CheckerBoardTexture : public Texture {
	//---------------------------
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
//egy interf?sz az objektumaink ?s a shaderek k?z?tt
//az objektumok ide teszik be minden olyan adatukat amit ?ta akarnak adni a shader-nek. A shader meg innen veszi ki ami neki kell
struct RenderState {
	//---------------------------
	mat4 MVP;		//model-view-projection mtx: az objektumot referenciahelyzetb?l egyb?l normaliz?lt eszk?zkoordin?tarendszerbe viszi (= M * V * P)
	mat4 M;			//modelez?si transzform: referenciahelyzetb?l vil?gkoordin?tarendszerbe visz ?t
	mat4 Minv;		//M inverze, a modelez?si transzform?ci? sor?n a norm?lvektorokra alkalmazzuk
	mat4 V;			//view mtx (kamer?ban defini?lva)
	mat4 P;			//projekci?s mtx (kamer?ban defini?lva)
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;		//egy Renderstate objektum v?ltoz?ival felt?lti az adott shader uniform v?ltoz?it

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
//t?bbf?le k?pp is megjelen?thetj?k az objektumainkat itt most 3 f?le k?pp van (GouraudShader, PhongShader ?s NPRShader)
//9. diasorban r?szletezve van (el?g lehet hf-ben a phong shader)
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec4 position_out;

		void main() {
			position_out = vec4(vtxPos, 1) * MVP;
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec4 position_out;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 ka = material.ka;
			vec3 kd = material.kd;

			//if(position_out.x < 2 && position_out.x > -2 && position_out.y < 2 && position_out.y > -2){
			//	kd = kd - 0.1f;
			//}
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	/// <summary>
	/// egy Renderstate objektum v?ltoz?ival felt?lti az adott shader uniform v?ltoz?it
	/// </summary>
	/// <param name="state"></param>
	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class Geometry {		//a geometriai objektumaink alaposzt?lya (9. diasor 8. dia)
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
//(9. diasor 9-10 dia)
class ParamSurface : public Geometry {
	//---------------------------
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x);		//u szerinti deriv?lt
		vec3 drdV(X.d.y, Y.d.y, Z.d.y);		//v szerinti deriv?lt
		vtxData.normal = cross(drdU, drdV);	//a norm?lvektor
		return vtxData;
	}

	//a gpu-ra felt?lti a parametrikus fel?let?nk(objektumunk) tesszell?lt v?ltozat?t (azaz h?romsz?gek sokas?g?t)
	//(9. diasor 10 dia)
	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

//---------------------------
class Square : public ParamSurface {
	//---------------------------
public:
	Square() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = U; Y = V; Z = Dnum2();
	}
};

//---------------------------
class Circle : public ParamSurface {
	//---------------------------
public:
	Circle() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Dnum2();
	}
};

Camera camera; // 3D camera
//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	vec3 velocity;
	bool isMoving;
	float mass;
	bool destroyed;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry, float _mass) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
		velocity = vec3(0.0f, 0.0f, 0.0f);
		isMoving = false;
		mass = _mass;
		destroyed = false;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		if (destroyed) return;
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend, vec3 _force) { 
		if (destroyed) return;
		//rotationAngle = 0.8f * tend; 
		float dt = tend - tstart;
		vec3 force = vec3(_force.x, _force.y, 0.0f);
		vec3 acceleration = force * (1 / mass);
		velocity = velocity + acceleration * dt;
		if(isMoving)
			MoveForward();
	}

	void MoveForward() {
		translation = translation + velocity;
		//printf("%f, %f, %f\n", translation.x, translation.y, translation.z);
		if (translation.x > (camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)) * camera.asp)) {
			translation.x = -(camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)) * camera.asp);
		}
		if (translation.x < -(camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)) * camera.asp)) {
			translation.x = (camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)) * camera.asp);
		}
		if (translation.y > (camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)))) {
			translation.y = -(camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)));
		}
		if (translation.y < -(camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)))) {
			translation.y = (camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)));
		}
	}
};

std::vector<Object*> objects;
Object* lastSphereObject;
const float fNewton = 0.003f;

//---------------------------
class Scene {
	//---------------------------
	std::vector<Light> lights;
	Shader* phongShader;
public:
	void Build() {
		// Shaders
		phongShader = new PhongShader();

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		// Textures
		Texture* texture4x8 = new CheckerBoardTexture(4, 8);
		Texture* texture15x20 = new CheckerBoardTexture(15, 20);	//k?s?bb t?rlend? !!!!!!!!!!!!!!!

		// Camera
		camera.wEye = vec3(0, 0, 15);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		Geometry* square = new Square();
		Object* squareObject1 = new Object(phongShader, material0, texture15x20, square, 0.0f);
		squareObject1->translation = vec3(-50.0f, -50.0f, -0.1f);
		squareObject1->scale = vec3(100.0f, 100.0f, 1.0f);
		objects.push_back(squareObject1);

		// Geometries
		Geometry* sphere = new Sphere();

		// Create objects by setting up their vertex data on the GPU
		Object* sphereObject1 = new Object(phongShader, material0, texture15x20, sphere, 50.0f);
		lastSphereObject = sphereObject1;
		float x = -(camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)) * camera.asp);
		float y = -(camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)));
		sphereObject1->translation = vec3(x + 0.5f, y + 0.5f, 0.15f);
		sphereObject1->scale = vec3(0.4f, 0.4f, 0.4f);
		objects.push_back(sphereObject1);

		// Lights
		lights.resize(2);
		lights[0].wLightPos = vec4(7, 4, 3, 1);	// not ideal point -> dot light source
		lights[0].La = vec3(0.1f, 0.1f, 1);
		lights[0].Le = vec3(2, 2, 2);

		//lights[1].wLightPos = vec4(5, 10, 20, 0);	// ideal point -> directional light source
		//lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		//lights[1].Le = vec3(1, 1, 1);

		lights[1].wLightPos = vec4(-5, -6, 3, 1);	// not ideal point -> dot light source
		lights[1].La = vec3(0.1f, 0.1f, 0.1f);
		lights[1].Le = vec3(2, 2, 2);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		for (Object* obj : objects) {
			vec3 force = vec3(0, 0, 0);
			if (dynamic_cast<Sphere*>(obj->geometry)) {
				for (Object* heavyObj : objects) {
					if (dynamic_cast<Circle*>(heavyObj->geometry)) {
						vec3 dr = heavyObj->translation - obj->translation;
						float dist = length(dr);
						if (dist < 0.35f) obj->destroyed = true;
						force = force + dr * (fNewton * obj->mass * heavyObj->mass / pow(dist, 2));
					}
				}
			}
			obj->Animate(tstart, tend, force);
		}
		AnimateLights(tstart, tend);
	}

	void AnimateLights(float tstart, float tend) {
		float dt = (tend - tstart) / 5;

		vec4 quaternion = Quaternion(dt, vec3(0, 0, 1));
		vec3 lightPos1 = vec3(lights[0].wLightPos.x, lights[0].wLightPos.y, lights[0].wLightPos.z);
		vec3 lightPos2 = vec3(lights[1].wLightPos.x, lights[1].wLightPos.y, lights[1].wLightPos.z);
		vec3 lightPos1Temp;
		vec3 lightPos2Temp;

		//calc new lightpos1
		lightPos1Temp = lightPos1 - lightPos2;
		lightPos1Temp = Rotate(lightPos1Temp, quaternion);
		lightPos1Temp = lightPos1Temp + lightPos2;


		//calc new lightpos1
		lightPos2Temp = lightPos2 - lightPos1;
		lightPos2Temp = Rotate(lightPos2Temp, quaternion);
		lightPos2Temp = lightPos2Temp + lightPos1;

		//setting both light positions
		lights[0].wLightPos = vec4(lightPos1Temp.x, lightPos1Temp.y, lightPos1Temp.z, 1);
		lights[1].wLightPos = vec4(lightPos2Temp.x, lightPos2Temp.y, lightPos2Temp.z, 1);
		//printf("%f, %f, %f, %f\n", lights[0].wLightPos.x, lights[0].wLightPos.y, lights[0].wLightPos.z, lights[0].wLightPos.w);
		//printf("%f, %f, %f, %f\n\n", lights[1].wLightPos.x, lights[1].wLightPos.y, lights[1].wLightPos.z, lights[1].wLightPos.w);
	}

	void CreateNewBall() {
		Material* material = new Material;
		material->kd = vec3(0.6f, 0.4f, 0.2f);
		material->ks = vec3(4, 4, 4);
		material->ka = vec3(0.1f, 0.1f, 0.1f);
		material->shininess = 30;

		Texture* texture15x20 = new CheckerBoardTexture(15, 20);		//k?s?bb t?rlend? !!!!!!!!!!!!!!!

		Geometry* sphere = new Sphere();

		Object* sphereObject1 = new Object(phongShader, material, texture15x20, sphere, 5.0f);
		lastSphereObject = sphereObject1;
		float x = -(camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)) * camera.asp);
		float y = -(camera.wEye.z / tan((camera.fovAngle * (float)M_PI / 180.0f)));
		sphereObject1->translation = vec3(x + 0.5f, y + 0.5f, 0.15f);
		sphereObject1->scale = vec3(0.4f, 0.4f, 0.4f);
		objects.push_back(sphereObject1);
	}

	void CreateNewCircle(float cX, float cY, float size) {
		Material* material2 = new Material;
		material2->kd = vec3(0.5f, 0.3f, 0.1f);
		material2->ks = vec3(4, 4, 4);
		material2->ka = vec3(0.1f, 0.1f, 0.1f);
		material2->shininess = 100;

		Texture* texture15x20 = new CheckerBoardTexture(15, 20);		//k?s?bb t?rlend? !!!!!!!!!!!!!!!

		float z = 0.01f;
		for (float i = 1.0f ; i < 5.0f ; i++)
		{
			Geometry* circle = new Circle();
			Object* circleObject1 = new Object(phongShader, material2, texture15x20, circle, size / 2.0f * 99.0f);
			float x = cX * ((camera.wEye.z - z) / tan((camera.fovAngle * (float)M_PI / 180.0f)) * camera.asp);
			float y = cY * ((camera.wEye.z - z) / tan((camera.fovAngle * (float)M_PI / 180.0f)));
			circleObject1->translation = vec3(x, y, z);
			circleObject1->scale = vec3(size / 2.0f / i, size / 2.0f / i, 1.0f);
			objects.push_back(circleObject1);
			z += 0.1f;
			Material* material2 = new Material;
			material2->kd = vec3(0.2f, 0.1f, 0.1f);
			material2->ks = vec3(4, 4, 4);
			material2->ka = vec3(0.1f, 0.1f, 0.1f);
			material2->shininess = 20;
		}
	}
	vec4 qmul(vec4 q1, vec4 q2) { // kvaterni? szorz?s
		vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
		vec3 tempVec3 = d2 * q1.w + d1 * q2.w + cross(d1, d2);
		return vec4(tempVec3.x, tempVec3.y, tempVec3.z, q1.w * q2.w - dot(d1, d2));
	}
	vec3 Rotate(vec3 u, vec4 q) {
		vec4 qinv(-q.x, -q.y, -q.z, q.w);
		vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
		return vec3(qr.x, qr.y, qr.z);
	}

	vec4 Quaternion(float ang, vec3 axis) {
		vec3 d = normalize(axis) * sinf(ang / 2);
		return vec4(d.x, d.y, d.z, cosf(ang / 2));
	}
};

Scene scene;
float size;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
	size = 2;
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { 
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis		//kisz?moljuk a normaliz?lt eszk?zkoordin?tarendszer beli kurzor poz?ci?t
		float cY = 1.0f - 2.0f * pY / windowHeight;
		vec2 velocity;
		velocity = vec2(cX, cY) - vec2(-1.0f, -1.0f);
		//printf("%f, %f", velocity.x, velocity.y);

		lastSphereObject->velocity = velocity * 0.1f;
		lastSphereObject->isMoving = true;

		scene.CreateNewBall();
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis		//kisz?moljuk a normaliz?lt eszk?zkoordin?tarendszer beli kurzor poz?ci?t
		float cY = 1.0f - 2.0f * pY / windowHeight;

		scene.CreateNewCircle(cX, cY, size++);
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ?infinitesimal?
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	float time = tend;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}