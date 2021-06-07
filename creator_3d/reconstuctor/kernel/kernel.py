from creator_3d.reconstuctor.kernel.pipeline import Pipeline


class Kernel:
    @staticmethod
    def run_pipeline(image_paths,
                     camera,
                     extractor_class, extractor_params,
                     matcher_class, matcher_params,
                     reconstructor_class, reconstructor_params,
                     bundle_adjuster_class, bundle_adjuster_params):

        # create step objects
        extractor = extractor_class(**extractor_params)
        matcher = matcher_class(**matcher_params)
        reconstructor = reconstructor_class(**reconstructor_params)
        bundle_adjuster = bundle_adjuster_class(**bundle_adjuster_params)

        pipeline = Pipeline(feature_extractor=extractor,
                            feature_matcher=matcher,
                            reconstructor=reconstructor,
                            bundle_adjuster=bundle_adjuster,
                            camera=camera)
        point_cloud = pipeline.run(image_paths)
        return point_cloud

