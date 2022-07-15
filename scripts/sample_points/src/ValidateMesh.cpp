// Copyright 2004-2019 Facebook. All Rights Reserved.
// Copyright 2020 Korrawe Karunratanakul. All Rights Reserved.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>
#include <cnpy.h>

#include "Utils.h"

extern pangolin::GlSlProgram GetShaderProgram();

void preprocessMesh(pangolin::Geometry &geom) {
  // linearize the object indices
  {
    int total_num_faces = 0;

    for (const auto &object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {

        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        total_num_faces += ibo.h;
      }
    }

    //      const int total_num_indices = total_num_faces * 3;
    pangolin::ManagedImage<uint8_t> new_buffer(3 * sizeof(uint32_t),
                                               total_num_faces);

    pangolin::Image<uint32_t> new_ibo =
        new_buffer.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3,
                                                          total_num_faces);

    int index = 0;

    for (const auto &object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {

        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        std::cout << "ibo.h " << ibo.h << std::endl;
        for (int i = 0; i < ibo.h; ++i) {
          
          // std::cout << ibo(i,0) << " " << ibo(i,1) << " " << ibo(i,2) << std::endl;
          // std::cout << ibo.Row(i) std::endl;
          new_ibo.Row(index).CopyFrom(ibo.Row(i));
          ++index;
        }
      }
    }

    geom.objects.clear();
    auto faces = geom.objects.emplace(std::string("mesh"),
                                      pangolin::Geometry::Element());

    faces->second.Reinitialise(3 * sizeof(uint32_t), total_num_faces);

    faces->second.CopyFrom(new_buffer);

    new_ibo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(
        0, 0, 3, total_num_faces);
    faces->second.attributes["vertex_indices"] = new_ibo;
  }
  // remove textures
  geom.textures.clear();
}

bool validate_point(pangolin::Geometry &geom,
                  pangolin::Image<uint32_t> modelFaces,
                  bool vis,
                  float max_dist,
                  float rejection_criteria_obs,
                  float rejection_criteria_tri,
                  std::vector<Eigen::Vector3f> &vertices_out,
                  std::vector<Eigen::Vector3f> &normals_out,
                  std::vector<std::size_t> &point_tri_ids_out,
                  bool isObject) {
  
  bool is_valid = true;
  std::string windowName;
  if (isObject) {
    windowName = "Main_Obj";
  } else {
    windowName = "Main_Hand";
  }
  if (vis)
    pangolin::CreateWindowAndBind(windowName, 640, 480);
  else
    pangolin::CreateWindowAndBind(windowName, 1, 1);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_DITHER);
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_LINE_SMOOTH);
  glDisable(GL_POLYGON_SMOOTH);
  glHint(GL_POINT_SMOOTH, GL_DONT_CARE);
  glHint(GL_LINE_SMOOTH, GL_DONT_CARE);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);
  glDisable(GL_MULTISAMPLE_ARB);
  glShadeModel(GL_FLAT);
  
  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, -max_dist,
                                             max_dist, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));
  pangolin::OpenGlRenderState s_cam2(
      pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, max_dist,
                                             -max_dist, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

  // Create Interactive View in window
  pangolin::Handler3D handler(s_cam);

  pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);

  pangolin::GlSlProgram prog = GetShaderProgram();

  if (vis) {
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit()) {
      // Clear screen and activate view to render into
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      d_cam.Activate(s_cam);

      prog.Bind();
      prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
      prog.SetUniform("V", s_cam.GetModelViewMatrix());

      pangolin::GlDraw(prog, gl_geom, nullptr);
      prog.Unbind();

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }
  // Create Framebuffer with attached textures
  size_t w = 400;
  size_t h = 400;
  pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
  pangolin::GlTexture normals(w, h, GL_RGBA32F);
  pangolin::GlTexture vertices(w, h, GL_RGBA32F);
  pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

  // View points around a sphere.
  std::vector<Eigen::Vector3f> views =
      EquiDistPointsOnSphere(100, max_dist * 1.1);

  std::vector<Eigen::Vector4f> point_normals;
  std::vector<Eigen::Vector4f> point_verts;
  std::vector<std::size_t> point_tri_ids; ///
  size_t num_tri = modelFaces.h;
  std::vector<Eigen::Vector4f> tri_id_normal_test(num_tri);
  for (size_t j = 0; j < num_tri; j++)
    tri_id_normal_test[j] = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
  int total_obs = 0;
  int wrong_obs = 0;
  for (unsigned int v = 0; v < views.size(); v++) {
    // change camera location
    s_cam2.SetModelViewMatrix(pangolin::ModelViewLookAt(
        views[v][0], views[v][1], views[v][2], 0, 0, 0, pangolin::AxisY));
    // Draw the scene to the framebuffer
    framebuffer.Bind();
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    prog.Bind();
    prog.SetUniform("MVP", s_cam2.GetProjectionModelViewMatrix());
    prog.SetUniform("V", s_cam2.GetModelViewMatrix());
    prog.SetUniform("ToWorld", s_cam2.GetModelViewMatrix().Inverse());
    prog.SetUniform("slant_thr", -1.0f, 1.0f);
    prog.SetUniform("ttt", 1.0, 0, 0, 1);
    pangolin::GlDraw(prog, gl_geom, nullptr);
    prog.Unbind();

    framebuffer.Unbind();
    pangolin::TypedImage img_normals;
    normals.Download(img_normals);
    std::vector<Eigen::Vector4f> im_norms = ValidPointsAndTrisFromIm(
        img_normals.UnsafeReinterpret<Eigen::Vector4f>(), tri_id_normal_test,
        point_tri_ids, 
        total_obs, wrong_obs);
    point_normals.insert(point_normals.end(), im_norms.begin(), im_norms.end());
    pangolin::TypedImage img_verts;
    vertices.Download(img_verts);
    std::vector<Eigen::Vector4f> im_verts =
        ValidPointsFromIm(img_verts.UnsafeReinterpret<Eigen::Vector4f>());
    point_verts.insert(point_verts.end(), im_verts.begin(), im_verts.end());
  }

  int bad_tri = 0;
  std::cout << "tri_id_normal_test.size() " << tri_id_normal_test.size() << std::endl;
  for (unsigned int t = 0; t < tri_id_normal_test.size(); t++) {
    if (tri_id_normal_test[t][3] < 0.0f)
      bad_tri++;
  }

  // std::cout << meshFileName << std::endl;
  std::cout << (float)(wrong_obs) / float(total_obs) << std::endl;
  std::cout << (float)(bad_tri) / float(num_tri) << std::endl;

  float wrong_ratio = (float)(wrong_obs) / float(total_obs);
  float bad_tri_ratio = (float)(bad_tri) / float(num_tri);

  if (wrong_ratio > rejection_criteria_obs ||
      bad_tri_ratio > rejection_criteria_tri) {
    is_valid = false;
    if (isObject) {
      std::cout << "mesh rejected object" << std::endl;
    } else {
      std::cout << "mesh rejected hand" << std::endl;
    }
  }

  std::vector<Eigen::Vector3f> vertices2;
  std::vector<Eigen::Vector3f> normals2;

  for (unsigned int v = 0; v < point_verts.size(); v++) {
    vertices2.push_back(point_verts[v].head<3>());
    normals2.push_back(point_normals[v].head<3>());
  }

  vertices_out = vertices2;
  normals_out = normals2;
  point_tri_ids_out = point_tri_ids;

  return is_valid;
}


int main(int argc, char **argv) {
  std::string meshFileName;
  std::string objFileName;
  std::string sample_id;
  std::string normalizationOutputFile;
  bool vis = false;

  std::string npyFileName;
  std::string npyObjFileName;
  std::string plyFileNameOut;
  std::string plyObjFileNameOut;
  std::string spatial_samples_npz;
  bool save_ply = true;
  bool test_flag = false;
  float variance = 0.005;
  // int num_sample = 500000;
  int num_sample = 20000; // 10000;
  float rejection_criteria_obs = 0.02f;
  float rejection_criteria_tri = 0.05f;
  float num_samp_near_surf_ratio = 1.f; //47.0f / 50.0f;

  CLI::App app{"PreprocessMesh"};
  app.add_option("--hand", meshFileName, "Hand Mesh File Name for Reading")->required();
  app.add_option("--obj", objFileName, "Object Mesh File Name for Reading")->required(); //
  app.add_option("--sample_id", sample_id, "sample id used for ofstream to .txt")->required(); //
  app.add_flag("-v", vis, "enable visualization");
  app.add_option("--ply", plyFileNameOut, "Save ply pc to here");
  app.add_option("--plyobj", plyObjFileNameOut, "Save ply for obj pc to here");
  app.add_option("-s", num_sample, "Save ply pc to here");
  app.add_option("--var", variance, "Set Variance");
  app.add_flag("--sply", save_ply, "save ply point cloud for visualization");
  app.add_flag("-t", test_flag, "test_flag");
  app.add_option("-n", spatial_samples_npz, "spatial samples from file");

  CLI11_PARSE(app, argc, argv);

  if (test_flag)
    variance = 0.05;

  float second_variance = variance / 10;
  std::cout << "variance: " << variance << " second: " << second_variance
            << std::endl;
  if (test_flag) {
    second_variance = variance / 100;
    num_samp_near_surf_ratio = 45.0f / 50.0f;
    num_sample = 250000;
  }

  std::cout << spatial_samples_npz << std::endl;

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

  // Hand
  pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);
  std::cout << geom.objects.size() << " hands" << std::endl;

  // Object
  pangolin::Geometry geomObj = pangolin::LoadGeometry(objFileName);
  std::cout << geomObj.objects.size() << " objects" << std::endl;

  //load into a new array
  cnpy::NpyArray arr = cnpy::npy_load("./hand_model_seg/face2label_sealed.npy");
  int8_t * loaded_data = arr.data<int8_t>();
  size_t nrows = arr.shape[0];
  size_t ncols = arr.shape[1];
  std::cout << "nrow " << nrows << std::endl;
  std::cout << "ncols " << ncols << std::endl; 

  preprocessMesh(geom);
  preprocessMesh(geomObj);

  pangolin::Image<uint32_t> modelFaces =
      pangolin::get<pangolin::Image<uint32_t>>(
          geom.objects.begin()->second.attributes["vertex_indices"]);

  pangolin::Image<uint32_t> modelFacesObj =
      pangolin::get<pangolin::Image<uint32_t>>(
          geomObj.objects.begin()->second.attributes["vertex_indices"]);
  
  // Compute min and max values for hand
  const std::pair<Eigen::Vector3f, Eigen::Vector3f> minMaxPoint =
        ComputeMinMax(geom);

  // Compute min and max values for object
  const std::pair<Eigen::Vector3f, Eigen::Vector3f> minMaxPointObj =
        ComputeMinMax(geomObj);

  // Calculate center point of hand and object
  float xMin, yMin, zMin, xMax, yMax, zMax; 
  xMin = fmin(minMaxPoint.first[0], minMaxPointObj.first[0]);
  yMin = fmin(minMaxPoint.first[1], minMaxPointObj.first[1]);
  zMin = fmin(minMaxPoint.first[2], minMaxPointObj.first[2]);
  xMax = fmax(minMaxPoint.second[0], minMaxPointObj.second[0]);
  yMax = fmax(minMaxPoint.second[1], minMaxPointObj.second[1]);
  zMax = fmax(minMaxPoint.second[1], minMaxPointObj.second[2]);

  const Eigen::Vector3f center((xMax + xMin) / 2.0f, (yMax + yMin) / 2.0f,
                               (zMax + zMin) / 2.0f);

  std::cout << "min x" << minMaxPoint.first[0] << ", min y " << minMaxPoint.first[1] << ", min z " << minMaxPoint.first[2] << std::endl;
  std::cout << "min x" << minMaxPointObj.first[0] << ", min y " << minMaxPointObj.first[1] << ", min z " << minMaxPointObj.first[2] << std::endl;

  float max_dist_hand = ComputeMaxDistance(geom, center);
  float max_dist_obj = ComputeMaxDistance(geomObj, center);

  float max_dist_before_normalize = fmax(max_dist_hand, max_dist_obj);
  float buffer_dist = 1.03;

  max_dist_before_normalize *= buffer_dist;

  std::cout << "max dist hand new function " << max_dist_hand << std::endl;
  std::cout << "max dist object new function " << max_dist_obj << std::endl;

  float max_dist = NormalizationWithParams(geom, center, max_dist_before_normalize, true);
  float max_dist_2 = NormalizationWithParams(geomObj, center, max_dist_before_normalize, true);

  std::cout << "max dist " << max_dist << std::endl;

  std::vector<Eigen::Vector3f> xyz_hand, xyz_obj;
  std::vector<float> sdf_hand, sdf_obj;
  std::vector<int> point_labels_hand, point_labels_obj;


  std::vector<Eigen::Vector3f> temp_vert;
  std::vector<Eigen::Vector3f> vertices_hand, vertices_obj;
  std::vector<Eigen::Vector3f> normals_hand, normals_obj;
  std::vector<std::size_t> point_tri_ids_hand, point_tri_ids_obj;

  // For object 
  bool valid_obj = validate_point(geomObj, modelFacesObj, vis,
               max_dist,
               rejection_criteria_obs,
               rejection_criteria_tri,
               vertices_obj, normals_obj, point_tri_ids_obj,
               true);
  if (!valid_obj){
      return 0;
  }

  //For hand
  bool valid_hand = validate_point(geom, modelFaces, vis,
               max_dist,
               rejection_criteria_obs,
               rejection_criteria_tri,
               vertices_hand, normals_hand, point_tri_ids_hand,
               false);
  if (!valid_hand){
      return 0;
  }

  std::ofstream outfile("valid_samples/" + sample_id + ".txt", std::ios_base::app);
  outfile<<sample_id<<std::endl;
  outfile.close();
  return 0;
}
